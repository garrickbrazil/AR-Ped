function [net_inputs, im_orig, valid] = get_rcnn_batch(conf, rois, phase, rpn_conf, rpn_net)
    
    training = strcmp(phase, 'train');
    
    if training
        set = conf.dataset_train;
        batch_size = conf.train_batch_size;
        im_dir = conf.train_dir;
    elseif strcmp(phase, 'val');
        set = conf.dataset_val; 
        batch_size = conf.test_batch_size;
        im_dir = conf.val_dir;
    else
        set = conf.dataset_test; 
        batch_size = conf.test_batch_size;
        im_dir = conf.test_dir;
        
        % special case for temporal_in
        if conf.use_temporal_in
            im_dir = strrep(conf.train_dir, '/train', '/test');
        end
        
    end

    rois_boxes = rois.boxes;
    valid = rois.scores>-1;
    valid = valid(1:batch_size);
    
    if conf.use_temporal_in
        
        im = (imread([im_dir '/images/' rois.image_id conf.ext]));
        im_orig = im;
        
        ims = repmat(im,[1 1 conf.temporal_num]);
        
        reg = regexp(rois.image_id, '(set\d\d_V\d\d\d_I)(\d\d\d\d\d)', 'tokens');
        setvname = reg{1}{1};
        iname = reg{1}{2};
        inum = str2num(iname);
        imnums = (inum - conf.temporal_db_step*conf.temporal_num + conf.temporal_db_step):conf.temporal_db_step:(inum - conf.temporal_db_step);
        
        for imind=1:length(imnums)
            temp_imid = sprintf('%s%05d', setvname, imnums(imind));
            impath = [im_dir '/images/' temp_imid conf.ext];
            
            if exist(impath, 'file')
                ims(:,:, 1+3*(imind-1):3+3*(imind-1)) = (imread(impath));
            end
        end
        
        impadH = round(size(ims,1)*conf.padfactor);
        impadW = round(size(ims,2)*conf.padfactor);
        ims = padarray(ims, [impadH impadW]);
        %stackedim = [ims(:,:,1:3) ims(:,:,4:6) ims(:,:,7:9)];
        %imshow(stackedim); drawnow;
        %imwrite(stackedim,['/home/gbmsu/Desktop/tmp/stackedims/' rois.image_id '.jpg']);
        rois_batch = single(zeros([conf.crop_size(2) conf.crop_size(1) size(ims,3) batch_size]));
    elseif isfield(conf, 'share_layer')
        
        im = (imread([im_dir '/images/' rois.image_id conf.ext]));
        im_orig = im;
        [~, ~, ~, ~, ~, ~, caffe_net] = proposal_im_detect(rpn_conf, rpn_net, im);
        share_layer = rpn_net.blobs(conf.share_layer).get_data();
        share_layer = permute(share_layer, [2, 1, 3, 4]);
        
        impadH = round(size(im,1)*conf.padfactor);
        impadW = round(size(im,2)*conf.padfactor);
        im = padarray(im, [impadH impadW]);
        
        sharepadH = round(size(share_layer,1)*conf.padfactor);
        sharepadW = round(size(share_layer,2)*conf.padfactor);
        share_layer = padarray(share_layer, [sharepadH sharepadW]);
        rois_batch = single(zeros([conf.crop_size(2) conf.crop_size(1) size(share_layer,3) batch_size]));
        
    else
        
        im = (imread([im_dir '/images/' rois.image_id conf.ext]));
        im_orig = im;
        impadH = round(size(im,1)*conf.padfactor);
        impadW = round(size(im,2)*conf.padfactor);
        im = padarray(im, [impadH impadW]);
        rois_batch = single(zeros([conf.crop_size(2) conf.crop_size(1) 3 batch_size]));
    end
    
    feat_scores_fg = rois.feat_scores_fg;
    feat_scores_bg = rois.feat_scores_bg;
    
    if training
        
        % paint weak seg gt
        if conf.has_weak
            
            ped_mask = uint8(zeros(size(im_orig,1), size(im_orig,2))); 
            
            for gtind=1:size(rois.gts,1)

                gt = rois.gts(gtind,:);
                
                x1 = min(max(round(gt(1)),1),size(ped_mask,2));
                y1 = min(max(round(gt(2)),1),size(ped_mask,1));
                x2 = min(max(round(gt(3)),1),size(ped_mask,2));
                y2 = min(max(round(gt(4)),1),size(ped_mask,1));

                ped_mask(y1:y2,x1:x2) = 1;

            end

            ped_mask = padarray(ped_mask, [impadH impadW]);
            
            weak_seg_batch = single(zeros([conf.weak_seg_crop(2) conf.weak_seg_crop(1) 1 batch_size])); 
            weak_seg_weights_batch = single(zeros([conf.weak_seg_crop(2) conf.weak_seg_crop(1) 1 batch_size]));
        end
        
        if conf.has_weak2
            weak_seg_batch2 = single(zeros([conf.weak_seg_crop(2)*2 conf.weak_seg_crop(1)*2 1 batch_size])); 
            weak_seg_weights_batch2 = single(zeros([conf.weak_seg_crop(2)*2 conf.weak_seg_crop(1)*2 1 batch_size]));
        end
        if conf.has_weak3
            weak_seg_batch3 = single(zeros([conf.weak_seg_crop(2)*4 conf.weak_seg_crop(1)*4 1 batch_size])); 
            weak_seg_weights_batch3 = single(zeros([conf.weak_seg_crop(2)*4 conf.weak_seg_crop(1)*4 1 batch_size]));
        end
    
        rois_labels   = rois.ols >= conf.fg_thresh;
        
        if isfield(conf, 'fg_thresh2')
            rois_labels2  = rois.ols >= conf.fg_thresh2;
        else
            rois_labels2  = rois.ols >= 0.5;
        end
        rois_ols     = rois.ols;
        rois_targets = rois.targets;
        cost_weights = zeros([batch_size,1]); 
    end
    
    for j=1:batch_size

        % get box info
        x1 = rois_boxes(j, 1);
        y1 = rois_boxes(j, 2);
        x2 = rois_boxes(j, 3);
        y2 = rois_boxes(j, 4);
        w = x2-x1;
        h = y2-y1;
        
        x1 = x1 - w*conf.padfactor + impadW;
        y1 = y1 - h*conf.padfactor + impadH;
        w = w + w*conf.padfactor;
        h = h + h*conf.padfactor;

        if isfield(conf, 'share_layer')
             % get box info
            shared_x1 = rois_boxes(j, 1)/conf.share_stride;
            shared_y1 = rois_boxes(j, 2)/conf.share_stride;
            shared_x2 = rois_boxes(j, 3)/conf.share_stride;
            shared_y2 = rois_boxes(j, 4)/conf.share_stride;
            %{
            shared_w = shared_x2-shared_x1/conf.share_stride;
            shared_h = shared_y2-shared_y1/conf.share_stride;
            
            shared_x1 = (shared_x1 - shared_w*conf.padfactor)/conf.share_stride + sharepadW;
            shared_y1 = (shared_y1 - shared_h*conf.padfactor)/conf.share_stride + sharepadH;
            shared_w  = (shared_w + shared_w*conf.padfactor)/conf.share_stride;
            shared_h  = (shared_h + shared_h*conf.padfactor)/conf.share_stride;
            shared_x2 = shared_x1 + shared_w;
            shared_y2 = shared_y1 + shared_h;
            %}
        end
        
        if conf.use_temporal_in
            stackedcrops = [];
            
            for imind=1:conf.temporal_num
            
                % crop and resize proposal
                propim = imcrop(ims(:,:,1+3*(imind-1):3+3*(imind-1)), [x1 y1 w h]);
                propim = imresize(single(propim), [conf.crop_size(1) conf.crop_size(2)]);

                if imind==1
                    stackedcrops = [propim];
                else
                    stackedcrops = [stackedcrops propim];
                end
                
                propim = bsxfun(@minus, single(propim), conf.image_means);

                % permute data into caffe c++ memory, thus [num, channels, height, width]
                propim = propim(:, :, [3, 2, 1], :);
                propim = permute(propim, [2, 1, 3, 4]);
                rois_batch(:,:,1+3*(imind-1):3+3*(imind-1),j) = single(propim);
            end
            %imshow(uint8(stackedcrops)); drawnow;
            %imwrite(stackedim,['/home/gbmsu/Desktop/tmp/stackedcrops/' rois.image_id '_' num2str(j) '.jpg']);
        elseif isfield(conf, 'share_layer')
            
            % crop and resize proposal
            maxX = size(share_layer,2);
            maxY = size(share_layer,1);
            
            shared_x1 = min(max(round(shared_x1), 1), maxX);
            shared_x2 = min(max(round(shared_x2), 1), maxX);
            
            shared_y1 = min(max(round(shared_y1), 1), maxY);
            shared_y2 = min(max(round(shared_y2), 1), maxY);
            
            propim = share_layer(shared_y1:shared_y2, shared_x1:shared_x2, :, :);
            propim = imresize(single(propim), [conf.crop_size(1) conf.crop_size(2)]);

            % permute data into caffe c++ memory, thus [num, channels, height, width]
            propim = permute(propim, [2, 1, 3, 4]);
            rois_batch(:,:,:,j) = single(propim);
        else
            
            % crop and resize proposal
            propim = imcrop(im, [x1 y1 w h]);
            propim = imresize(single(propim), [conf.crop_size(1) conf.crop_size(2)]);
            propim = bsxfun(@minus, single(propim), conf.image_means);

            % permute data into caffe c++ memory, thus [num, channels, height, width]
            propim = propim(:, :, [3, 2, 1], :);
            propim = permute(propim, [2, 1, 3, 4]);
            rois_batch(:,:,:,j) = single(propim);
        end
        
        if training 
        
            cost_weights(j) = h/conf.cost_mean_height; 
            
            if conf.has_weak
                prop_mask = imcrop(ped_mask, [x1 y1 w h]);
                prop_mask = imresize(single(prop_mask), [conf.weak_seg_crop(1) conf.weak_seg_crop(2)], 'nearest');
                prop_mask = permute(prop_mask, [2, 1, 3, 4]);
                weak_seg_batch(:,:,:,j) = single(prop_mask);
            end
            
            if conf.has_weak2
                prop_mask = imcrop(ped_mask, [x1 y1 w h]);
                prop_mask = imresize(single(prop_mask), [conf.weak_seg_crop(1)*2 conf.weak_seg_crop(2)*2], 'nearest');
                prop_mask = permute(prop_mask, [2, 1, 3, 4]);
                weak_seg_batch2(:,:,:,j) = single(prop_mask);
            end
            
            if conf.has_weak3
                prop_mask = imcrop(ped_mask, [x1 y1 w h]);
                prop_mask = imresize(single(prop_mask), [conf.weak_seg_crop(1)*4 conf.weak_seg_crop(2)*4], 'nearest');
                prop_mask = permute(prop_mask, [2, 1, 3, 4]);
                weak_seg_batch3(:,:,:,j) = single(prop_mask);
            end
            
        end
        
    end
    
    if training
        
        rois_labels  = single(rois_labels(1:batch_size));
        rois_labels2 = single(rois_labels2(1:batch_size));
        rois_targets = single(rois_targets(1:batch_size));
        rois_ols     = single(rois_ols(1:batch_size));
        rois_label_weights = ones([size(rois_labels),1]);
        
        if conf.use_best && ~isempty(rois.ignores)
            ugts = unique(rois_targets);
            for ugt=ugts'

                if ~rois.ignores(ugt)
                    tar_ols = rois_ols .* (rois_targets==ugt);
                    [ugtval, ugtind] = max(tar_ols);
                    if ugtval>=0.5
                        rois_labels(ugtind) = 1;
                    end
                else
                    tar_ols = rois_ols .* (rois_targets==ugt);
                    [ugtval, ugtind] = max(tar_ols);
                    if ugtval>=0.5
                        rois_labels(ugtind) = 1;
                    end
                end
            end
        end
        
        if conf.cost_sensitive
            rois_label_weights = rois_label_weights + cost_weights;
        end
        
        if safeDefault(conf, 'use_best_only') && ~isempty(rois.ignores)
            rois_labels = rois_labels*0;
            
            ugts = unique(rois_targets);
            for ugt=ugts'

                if ~rois.ignores(ugt)
                    tar_ols = rois_ols .* (rois_targets==ugt);
                    [ugtval, ugtind] = max(tar_ols);
                    if ugtval>=0.5
                        rois_labels(ugtind) = 1;
                    end
                else
                    tar_ols = rois_ols .* (rois_targets==ugt);
                    [ugtval, ugtind] = max(tar_ols);
                    if ugtval>=0.5
                        rois_labels(ugtind) = 1;
                    end
                end
                
            end
        end
        
        best_labels = rois_labels*0;
        if conf.attach_best && ~isempty(rois.ignores)
            ugts = unique(rois_targets);
            for ugt=ugts'

                if ~rois.ignores(ugt)
                    tar_ols = rois_ols .* (rois_targets==ugt);
                    [ugtval, ugtind] = max(tar_ols);
                    if ugtval>=0.5
                        best_labels(ugtind) = 1;
                    end
                else
                    tar_ols = rois_ols .* (rois_targets==ugt);
                    [ugtval, ugtind] = max(tar_ols);
                    if ugtval>=0.5
                        best_labels(ugtind) = 1;
                    end
                end
            end
        end

        rois_fg = rois_labels == 1;
                
        if ~conf.natural_fg_weight && sum(rois_fg(:)) > 0, 
            if isfield(conf, 'fg_weight')
                fg_weight = conf.fg_weight;
            else
                fg_count = sum(rois_fg(:));
                bg_count = sum(~rois_fg(:));
                fg_weight = conf.fg_fraction*bg_count/fg_count;
            end
            rois_label_weights(rois_fg) = fg_weight;
        end
        
        if safeDefault('conf', 'stage1_importance_v1')
            
            for roiind=1:length(rois_label_weights)
                
                roilabel = rois_labels(roiind);
                
                if roilabel==0
                   roiscore = exp(rois.feat_scores_bg(roiind))/(exp(rois.feat_scores_fg(roiind)) + exp(rois.feat_scores_bg(roiind)));   
                else
                   roiscore = exp(rois.feat_scores_fg(roiind))/(exp(rois.feat_scores_fg(roiind)) + exp(rois.feat_scores_bg(roiind)));
                end
                roiweight = (1 - roiscore) + eps;
                rois_label_weights(roiind) = rois_label_weights(roiind) + roiweight;
            end
            
        end
        
        if safeDefault('conf', 'stage1_importance_v2')
            
            for roiind=1:length(rois_label_weights)
                
                roilabel = rois_labels(roiind);
                
                if roilabel==0
                   roiscore = exp(rois.feat_scores_bg(roiind))/(exp(rois.feat_scores_fg(roiind)) + exp(rois.feat_scores_bg(roiind)));   
                else
                   roiscore = exp(rois.feat_scores_fg(roiind))/(exp(rois.feat_scores_fg(roiind)) + exp(rois.feat_scores_bg(roiind)));
                end
                roiweight = roiscore + eps;
                rois_label_weights(roiind) = rois_label_weights(roiind)/roiweight;
            end
            
        end
        
        rois_labels  = single(permute(rois_labels, [3, 4, 2, 1]));
        rois_labels2 = single(permute(rois_labels2, [3, 4, 2, 1]));
        best_labels  = single(permute(best_labels, [3, 4, 2, 1]));
        rois_label_weights = single(permute(rois_label_weights, [3, 4, 2, 1]));
        net_inputs = {rois_batch, rois_labels, rois_label_weights};
       
        rois_bbox         = zeros(length(rois_labels), 4, 'single');
        rois_bbox_weights = zeros(length(rois_labels), 4, 'single');
        
        if training && safeDefault(conf, 'has_bbox_reg')
            
            if ~isempty(rois.gts)
                for bind=1:length(rois_labels)


                    box = rois_boxes(bind, :);
                    tar = rois_targets(bind);
                    gt  = rois.gts(tar, :);
                    ign = rois.ignores(tar);
                    ol = rois_ols(bind);

                    if ~ign && ol>0.5

                        x1 = box(1);
                        y1 = box(2);
                        x2 = box(3);
                        y2 = box(4);

                        cx = (x1 + x2)/2;
                        cy = (y1 + y2)/2;

                        w = (x2-x1)+1;
                        h = (y2-y1)+1;

                        gt_x1  = gt(1);
                        gt_y1  = gt(2);
                        gt_x2  = gt(3);
                        gt_y2  = gt(4);

                        gt_w  = gt_x2-gt_x1+1;
                        gt_h  = gt_y2-gt_y1+1;

                        gt_cx  = (gt_x2 + gt_x1)/2;
                        gt_cy  = (gt_y2 + gt_y1)/2;

                        % shift xy
                        dx = gt_cx - cx;
                        dy = gt_cy - cy; 
                        dx = dx/w;
                        dy = dy/h;

                        % scale wh
                        dw = w/gt_w;
                        dh = h/gt_h;

                        dw = log(dw);
                        dh = log(dh);

                        assert(all(abs([dx dy dw dh]) <= 1));

                        % apply to box
                        new_cx = cx + dx*w;
                        new_cy = cy + dy*h;
                        new_w  = w/exp(dw);
                        new_h  = h/exp(dh);
                        new_x1 = (2*new_cx - new_w + 1)/2;
                        new_y1 = (2*new_cy - new_h + 1)/2;
                        new_x2 = new_x1 + new_w - 1;
                        new_y2 = new_y1 + new_h - 1;

                        rois_bbox(bind,:)          = [dx, dy, dw, dh];
                        rois_bbox_weights(bind,:)  = rois_label_weights(bind);

                    end
                end
            end
            
            net_inputs{end+1}  = single(permute(rois_bbox, [3, 4, 2, 1]));
            net_inputs{end+1}  = single(permute(rois_bbox_weights, [3, 4, 2, 1]));
        end
        
        if conf.attach_best
            net_inputs{end+1} = best_labels;
        end
        
        if isfield(conf, 'attach_fg2') && conf.attach_fg2
            net_inputs{end+1} = rois_labels2;
        end
        
        if conf.has_weak
            
            for weakind=1:batch_size
                weak_seg_weights_batch(:,:,:,weakind) = rois_label_weights(weakind);
                
                if conf.has_weak2
                    weak_seg_weights_batch2(:,:,:,weakind) = rois_label_weights(weakind);
                end
                if conf.has_weak3
                    weak_seg_weights_batch3(:,:,:,weakind) = rois_label_weights(weakind);
                end
            end
            
            net_inputs{length(net_inputs) + 1} = weak_seg_batch;
            net_inputs{length(net_inputs) + 1} = weak_seg_weights_batch;
            
            if conf.has_weak2
                net_inputs{length(net_inputs) + 1} = weak_seg_batch2;
                net_inputs{length(net_inputs) + 1} = weak_seg_weights_batch2;
            end
            if conf.has_weak3
                net_inputs{length(net_inputs) + 1} = weak_seg_batch3;
                net_inputs{length(net_inputs) + 1} = weak_seg_weights_batch3;
            end
            
        end
        
    % testing
    else
        net_inputs = {rois_batch};
    end
    
    if training && conf.contrastive
       
        combs = combnk(1:batch_size, 2);
        pair_labels = single(zeros(size(combs,1),1));
        
        for pairind=1:size(combs,1)
            
            f1 = combs(pairind,1);
            f2 = combs(pairind,2);
            
            ol1 = rois_ols(f1);
            ol2 = rois_ols(f2);
            t1 = rois_targets(f1);
            t2 = rois_targets(f2);
            
            if ol1 < 0.3 || ol2 < 0.3
                pair_labels(pairind) = -1;
            else
                pair_labels(pairind) = t1==t2;
            end
        end
        
        targets = single(permute(rois_targets(1:batch_size), [3, 4, 2, 1]));
        ols = single(permute(rois_ols(1:batch_size), [3, 4, 2, 1]));
        pair_labels = single(permute(pair_labels, [3, 4, 2, 1]));
        net_inputs{length(net_inputs) + 1} = pair_labels;
        net_inputs{length(net_inputs) + 1} = targets;
        net_inputs{length(net_inputs) + 1} = ols;
    end

    if safeDefault(conf, 'concat_radial_grad')
        auto_data = rois.auto_data;
        auto_data = auto_data(:,:,:,1:batch_size);
        auto_data = single(permute(auto_data, [2, 1, 3, 4]));
        net_inputs{end+1} = auto_data;
    end
    
    if conf.feat_scores
        rois_feat_scores = [feat_scores_bg feat_scores_fg];
        rois_feat_scores = single(rois_feat_scores(1:batch_size, :));
        rois_feat_scores = single(permute(rois_feat_scores, [3, 4, 2, 1]));
        net_inputs{length(net_inputs) + 1} = rois_feat_scores;
    end
    
    if safeDefault(conf, 'hard_thres')
        
        rois_scores = rois.scores(1:batch_size);
        valid = rois_scores>=conf.hard_thres;
        
        for netind=1:length(net_inputs)
            bdata = net_inputs{netind};
            bdata = bdata(:,:,:,valid);
            net_inputs{netind} = bdata;
        end
    end

end