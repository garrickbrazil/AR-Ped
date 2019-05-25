function [input_blobs, random_scale_inds, im_rgb, im_aug] = proposal_generate_minibatch(conf, image_roidb, rpn_test_net)
% [input_blobs, random_scale_inds, im_rgb] = proposal_generate_minibatch(conf, image_roidb)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

    input_blobs = {};
    
    num_images = length(image_roidb);
    %assert(num_images == 1, 'only support num_images == 1');

    % Sample random scales to use for each image in this batch
    random_scale_inds = randi(length(conf.scales), num_images, 1);
    
    % Get the input image blob
    [im_blob, im_scales, im_rgb, im_aug] = get_image_blob(conf, image_roidb, random_scale_inds);
    
    for bind=1:num_images
        
        starting_input = length(input_blobs);
        
        rois = image_roidb(bind);

        boxes = rois.boxes;

        % get fcn output size
        img_size = round(image_roidb(bind).im_size * im_scales(1));
        output_size = [calc_output_size(img_size(1), conf), calc_output_size(img_size(2), conf)];

        if conf.mirror && size(boxes,1)>0

            im_w = rois.im_size(2);

            for bind=1:size(boxes,1)

                y1 = boxes(bind,2);
                y2 = boxes(bind,4);
                h = (y2-y1)+1;

                x1 = boxes(bind,1);
                x2 = boxes(bind,3);
                w = (x2-x1)+1;

                x1 = im_w - x1 - w;
                x2 = x1 + w - 1;

                boxes(bind,1) = x1;
                boxes(bind,3) = x2;

                %im_rgb = insertShape(im_rgb, 'rectangle', [x1 boxes(bind,2) w h]);

            end

            %imshow(im_rgb);
        end

        % weak segmentation
        if conf.has_weak 

            weak_masks        = {};
            weak_mask_weights = {};

            ped_mask_weights = single(ones(size(im_blob,1), size(im_blob,2)));
            ped_mask = uint8(zeros(size(im_blob,1), size(im_blob,2)));
            
            for gtind=1:size(boxes,1)

                ignore = rois.gt_ignores(gtind);
                gt = boxes(gtind,:);

                x1 = min(max(round(gt(1)*im_scales(1)),1),size(ped_mask,2));
                y1 = min(max(round(gt(2)*im_scales(1)),1),size(ped_mask,1));
                x2 = min(max(round(gt(3)*im_scales(1)),1),size(ped_mask,2));
                y2 = min(max(round(gt(4)*im_scales(1)),1),size(ped_mask,1));

                w = x2 - x1;
                h = y2 - y1;

                %if ~rois.igncls(gtind)
                    % assign fg label
                    ped_mask(y1:y2,x1:x2) = 1;
                %else
                    ped_mask(y1:y2,x1:x2) = 1; % changed gb
                %end

                % cost sensitive
                if conf.cost_sensitive, ped_mask_weights(y1:y2,x1:x2) = single(1 + h/(conf.cost_mean_height*im_scales(1))); end

            end

            for strideind=1:length(conf.weak_strides)

                stride = conf.weak_strides(strideind);

                ped_mask_tmp = imresize(single(ped_mask), 1/stride, 'nearest');
                ped_mask_weights_tmp = imresize(single(ped_mask_weights), 1/stride, 'nearest');
                ped_mask_tmp = permute(ped_mask_tmp, [2, 1, 3, 4]);
                ped_mask_weights_tmp = permute(ped_mask_weights_tmp, [2, 1, 3, 4]);

                weak_masks{strideind}        = ped_mask_tmp;
                weak_mask_weights{strideind} = ped_mask_weights_tmp;
            end
        end

        % dynamic IoU on?
        if safeDefault(conf, 'dynamic_iou') && conf.iter>conf.dynamic_start && exist('rpn_test_net', 'var')
            
            bbox_pred_layer = 'proposal_bbox_pred';
            cls_pred_layer  = 'proposal_cls_prob';

            scores_unfiltered = rpn_test_net.blobs(cls_pred_layer).get_data();
            scores_unfiltered = scores_unfiltered(:, :, end);
            scores_unfiltered = reshape(scores_unfiltered, size(rpn_test_net.blobs(bbox_pred_layer).get_data(), 1), size(rpn_test_net.blobs(bbox_pred_layer).get_data(), 2), []);
            scores_unfiltered = permute(scores_unfiltered, [3, 2, 1]);
            scores_unfiltered = scores_unfiltered(:);
        end
        
        %subplot(1,3,1); imshow(im_rgb); subplot(1,3,2); imshow(ped_mask==1);  subplot(1,3,3); imshow(ped_mask==255);
        %drawnow;

        if conf.split_anchors
            anchor_sets = conf.anchor_sets;
        else
            anchor_sets = conf.anchor_scales;
        end

        for aind=1:size(anchor_sets,1)

            if conf.split_anchors
                conf.anchor_scales = anchor_sets(aind,:);
                conf.anchors = proposal_generate_anchors(conf, true);
                conf.feat_stride = conf.anchor_sets_stride(aind);
            end

            output_size = [calc_output_size(img_size(1), conf), calc_output_size(img_size(2), conf)];

            maxrois = prod(output_size)*size(conf.anchors, 1);
            rois_per_image = min(conf.batch_size, maxrois) / 1;
            fg_rois_per_image = round(min(conf.batch_size, maxrois) * conf.fg_fraction);

            % init blobs
            labels_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1), 1);
            label_weights_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1), 1);
            bbox_targets_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1)*4, 1);
            bbox_loss_blob = zeros(output_size(2), output_size(1), size(conf.anchors, 1)*4, 1);

            if conf.use_recursive && conf.recursive_boxes && conf.iter > conf.recursive_start

                [labels, label_weights, bbox_targets, bbox_loss, labels2, label_weights2, labels3, label_weights3] = ...
                    sample_rois(conf, image_roidb(bind), fg_rois_per_image, rois_per_image, im_scales(1), scores_unfiltered);
            elseif safeDefault(conf, 'dynamic_iou') && conf.iter>conf.dynamic_start && exist('rpn_test_net', 'var')
                
                [labels, label_weights, bbox_targets, bbox_loss, labels2, label_weights2, labels3, label_weights3] = ...
                    sample_rois(conf, image_roidb(bind), fg_rois_per_image, rois_per_image, im_scales(1), scores_unfiltered);
            else
                [labels, label_weights, bbox_targets, bbox_loss, labels2, label_weights2, labels3, label_weights3] = ...
                    sample_rois(conf, image_roidb(bind), fg_rois_per_image, rois_per_image, im_scales(1));
            end

            assert(img_size(1) == size(im_blob, 1) && img_size(2) == size(im_blob, 2));

            cur_labels_blob = reshape(labels, size(conf.anchors, 1), output_size(1), output_size(2));
            cur_label_weights_blob = reshape(label_weights, size(conf.anchors, 1), output_size(1), output_size(2));
            cur_bbox_targets_blob = reshape(bbox_targets', size(conf.anchors, 1)*4, output_size(1), output_size(2));
            cur_bbox_loss_blob = reshape(bbox_loss', size(conf.anchors, 1)*4, output_size(1), output_size(2));

            labels_blob2 = zeros(output_size(2), output_size(1), size(conf.anchors, 1), 1);
            label_weights_blob2 = zeros(output_size(2), output_size(1), size(conf.anchors, 1), 1);
            labels_blob3 = zeros(output_size(2), output_size(1), size(conf.anchors, 1), 1);
            label_weights_blob3 = zeros(output_size(2), output_size(1), size(conf.anchors, 1), 1);

            cur_labels_blob2 = reshape(labels2, size(conf.anchors, 1), output_size(1), output_size(2));
            cur_label_weights_blob2 = reshape(label_weights2, size(conf.anchors, 1), output_size(1), output_size(2));
            cur_labels_blob3 = reshape(labels3, size(conf.anchors, 1), output_size(1), output_size(2));
            cur_label_weights_blob3 = reshape(label_weights3, size(conf.anchors, 1), output_size(1), output_size(2));

            cur_labels_blob2 = permute(cur_labels_blob2, [3, 2, 1]);
            cur_label_weights_blob2 = permute(cur_label_weights_blob2, [3, 2, 1]);
            cur_labels_blob3 = permute(cur_labels_blob3, [3, 2, 1]);
            cur_label_weights_blob3 = permute(cur_label_weights_blob3, [3, 2, 1]);

            % permute from [channel, height, width], where channel is the
            % fastest dimension to [width, height, channel]
            cur_labels_blob = permute(cur_labels_blob, [3, 2, 1]);
            cur_label_weights_blob = permute(cur_label_weights_blob, [3, 2, 1]);
            cur_bbox_targets_blob = permute(cur_bbox_targets_blob, [3, 2, 1]);
            cur_bbox_loss_blob = permute(cur_bbox_loss_blob, [3, 2, 1]);

            labels_blob2(:, :, :, 1) = single(cur_labels_blob2);
            label_weights_blob2(:, :, :, 1) = single(cur_label_weights_blob2);
            labels_blob3(:, :, :, 1) = single(cur_labels_blob3);
            label_weights_blob3(:, :, :, 1) = single(cur_label_weights_blob3);

            labels_blob(:, :, :, 1) = cur_labels_blob;

            label_weights_blob(:, :, :, 1) = cur_label_weights_blob;
            bbox_targets_blob(:, :, :, 1) = cur_bbox_targets_blob;
            bbox_loss_blob(:, :, :, 1) = cur_bbox_loss_blob;

            % permute data into caffe c++ memory, thus [num, channels, height, width]
            labels_blob = single(labels_blob);
            label_weights_blob = single(label_weights_blob);
            bbox_targets_blob = single(bbox_targets_blob); 
            bbox_loss_blob = single(bbox_loss_blob);

            assert(~isempty(im_blob));
            assert(~isempty(labels_blob));
            assert(~isempty(label_weights_blob));
            assert(~isempty(bbox_targets_blob));
            assert(~isempty(bbox_loss_blob));

            input_blobs{end+1} = labels_blob;
            input_blobs{end+1} = label_weights_blob;
            input_blobs{end+1} = bbox_targets_blob;
            input_blobs{end+1} = bbox_loss_blob;

            if safeDefault(conf, 'has_nms_lstm')
                input_blobs{end+1} = labels_blob2;
                input_blobs{end+1} = label_weights_blob2;
                input_blobs{end+1} = labels_blob3;
                input_blobs{end+1} = label_weights_blob3;
            end

            if safeDefault(conf, 'attach_fg2')
                input_blobs{end+1} = labels_blob2;
                input_blobs{end+1} = label_weights_blob2;
            end
        end

        if safeDefault(conf, 'attach_best')
            input_blobs{end+1} = labels_blob3;
            input_blobs{end+1} = label_weights_blob3;
        end
        
        if conf.split_anchors
            conf.anchor_scales = conf.orig_anchor_scales;
            conf.anchors       = conf.orig_anchors;
            conf.feat_stride   = conf.orig_feat_stride;
        end

        im_blob_mod = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
        im_blob_mod = single(permute(im_blob_mod, [2, 1, 3, 4]));
        im_blob_mod = im_blob_mod(:,:,:,bind);
        input_blobs = {input_blobs{1:starting_input} im_blob_mod input_blobs{(starting_input+1):end}};

        if conf.has_weak

            for strideind=1:length(conf.weak_strides)

                input_blobs{end+1} = weak_masks{strideind};
                input_blobs{end+1} = weak_mask_weights{strideind};
            end
        end
    end
    
    if num_images>1
        
        % stack batches more logically
        num_input_layers = length(input_blobs)/num_images;

        for lind=1:length(input_blobs)

            if lind>num_input_layers
                layer_ind = mod(lind-1, num_input_layers) + 1;
                batch_ind = floor((lind-1)/(num_input_layers)) + 1;
                input_blobs{layer_ind}(:,:,:,batch_ind) = input_blobs{lind};
            end
        end
        input_blobs = {input_blobs{1:num_input_layers}};
    end
end

%% Build an input blob from the images in the roidb at the specified scales.
function [im_blob, im_scales, im_, im_aug] = get_image_blob(conf, images, random_scale_inds)
    
    num_images = length(images);
    processed_ims = cell(num_images, 1);
    im_scales = nan(num_images, 1);
    for i = 1:num_images
        im = imread(images(i).image_path);
        
        if conf.mirror
            im = flip(im ,2);
        end
        
        im_ = im;
        
        im = augment_data(im, conf);
        im_aug = im;
        
        target_size = conf.scales(random_scale_inds(i));
        
        [im, im_scale] = prep_im_for_blob(im, conf.image_means, target_size, conf.max_size);
        
        im_scales(i) = im_scale;
        processed_ims{i} = im; 
    end
    
    im_blob = im_list_to_blob(processed_ims);
end

%% Generate a random sample of ROIs comprising foreground and background examples.
function [labels, label_weights, bbox_targets, bbox_loss_weights, labels2, label_weights2, labels3, label_weights3] = sample_rois(conf, image_roidb, fg_rois_per_image, rois_per_image, im_scale, pred)

    [anchors, ~] = proposal_locate_anchors(conf, image_roidb.im_size);

    gt_ignores = image_roidb.gt_ignores;
    
    img_size = round(image_roidb.im_size * im_scale);
    output_size = [calc_output_size(img_size(1), conf), calc_output_size(img_size(2), conf)];
    num_anchors = length(conf.anchors);
    
    boxes = image_roidb.boxes;
    
    if conf.mirror && size(boxes,1)>0
       
        im_w = image_roidb.im_size(2);
        
        for bind=1:size(boxes,1)
           
            y1 = boxes(bind,2);
            y2 = boxes(bind,4);
            h = (y2-y1)+1;
            
            x1 = boxes(bind,1);
            x2 = boxes(bind,3);
            w = (x2-x1)+1;
            
            x1 = im_w - x1 - w;
            x2 = x1 + w - 1;
            
            boxes(bind,1) = x1;
            boxes(bind,3) = x2;
            
            %im_rgb = insertShape(im_rgb, 'rectangle', [x1 boxes(bind,2) w h]);
            
        end
        
        %imshow(im_rgb);
    end
    
    % add by zhangll, whether the gt_rois empty?
    if isempty(boxes)

       [bbox_targets, overlaps, targets] = ...
           proposal_compute_targets(conf, boxes, gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale);
    else
       [bbox_targets, overlaps, targets] = ...
           proposal_compute_targets(conf, scale_rois(boxes, image_roidb.im_size, im_scale), gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale);
    end

    gt_inds = find(bbox_targets(:, 1) > 0);
    if ~isempty(gt_inds)
        bbox_targets(gt_inds, 2:end) = ...
            bsxfun(@minus, bbox_targets(gt_inds, 2:end), conf.bbox_means);
        bbox_targets(gt_inds, 2:end) = ...
            bsxfun(@rdivide, bbox_targets(gt_inds, 2:end), conf.bbox_stds);
    end
    
    ex_asign_labels = bbox_targets(:, 1);
    
    labels_fg = bbox_targets(:, 1) > 0;
    labels_bg = bbox_targets(:, 1) < 0;
    
    if safeDefault(conf, 'dynamic_iou') && conf.iter>conf.dynamic_start && exist('pred', 'var')
        
        step = 0.005;
        thr_bg  = conf.dynamic_iou_thresh;
        lbls = pred>thr_bg & labels_fg;
        
        % fill holes
        lbls = reshape(permute(imfill(permute(reshape(full(lbls), num_anchors, output_size(1), output_size(2)), [3 2 1]), 'holes'), [3 2 1]), [], 1);
        
        while length(unique(targets(labels_fg)))>length(unique(targets(lbls)))
            thr_bg = thr_bg - step;
            lbls = pred>thr_bg & labels_fg;
            
            % fill holes
            lbls = reshape(permute(imfill(permute(reshape(full(lbls), num_anchors, output_size(1), output_size(2)), [3 2 1]), 'holes'), [3 2 1]), [], 1);
        end
        
        labels_fg_orig = labels_fg;
        
        fg_to_fg = labels_fg_orig & lbls;
        fg_to_bg = labels_fg_orig & ~lbls;
        num_converted = 0+sum(fg_to_bg(:));
        
        ex_asign_labels(fg_to_bg) = 0;        % remove some fg
        labels_fg = labels_fg & fg_to_fg;     % remove some fg
        labels_bg = labels_bg | fg_to_bg;     % add some bg
        
    end
    
    % update labels "recursively" (based on predictions)
    if conf.use_recursive && conf.recursive_boxes && conf.iter > conf.recursive_start
    
        thr = conf.recursive_boxes_thr;
        lbls = pred>thr & labels_fg;
        
        % fill holes
        lbls = reshape(permute(imfill(permute(reshape(full(lbls), 9, 45, 60), [3 2 1]), 'holes'), [3 2 1]), [], 1);
        
        while length(unique(targets(labels_fg)))>length(unique(targets(lbls)))
            thr = thr - 0.001;
            lbls = pred>thr & labels_fg;
            
            % fill holes
            lbls = reshape(permute(imfill(permute(reshape(full(lbls), 9, 45, 60), [3 2 1]), 'holes'), [3 2 1]), [], 1);
        end
        
        labels_fg_orig = labels_fg;
        %labels_bg_orig = labels_bg;
        
        fg_to_fg = labels_fg_orig & lbls;
        fg_to_bg = labels_fg_orig & ~lbls;
        
        ex_asign_labels(fg_to_bg) = 0;        % remove some fg
        labels_fg = labels_fg & fg_to_fg;     % remove some fg
        labels_bg = labels_bg | fg_to_bg;     % add some bg
        
    end
    
    % Select foreground ROIs as those with >= FG_THRESH overlap
    fg_inds = find(labels_fg);

    % Select background ROIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = find(labels_bg);
    
    bbox_targets_orig = bbox_targets;
    
    % use best gt only
    % find the best box for each target GT
    % then add remaining fg -> bg
    % but do not touch any "ignore" regions
    if safeDefault(conf, 'best_gt_only')

        if length(gt_ignores)>0
            best_fg_inds = [];
            tars = unique(targets);

            for tarind=1:length(tars)

                tar = tars(tarind);
                if ~gt_ignores(tarind)
                    ols = overlaps.*(targets==tar);
                    [vals, inds] = sort(ols, 'descend');
                    inds = inds(vals>0);
                    %[val, ind] = max(ols);
                    num_gts_to_add = min(length(inds), conf.best_gt_num);
                    best_fg_inds = [best_fg_inds inds(1:num_gts_to_add)];
                end
            end

           % add all non-best to bg_inds
           bg_inds = [setdiff(bg_inds, best_fg_inds); setdiff(fg_inds, best_fg_inds)];
           fg_inds = best_fg_inds;
        end
    end
    
    % set foreground labels
    labels = zeros(size(bbox_targets, 1), 1);
    
    if safeDefault(conf, 'best_gt_only')
        labels(fg_inds) = 1;
    else
        labels(fg_inds) = ex_asign_labels(fg_inds);
        assert(all(ex_asign_labels(fg_inds) > 0));
    end
    
    % select foreground
    fg_num = min(fg_rois_per_image, length(fg_inds));
    fg_inds = fg_inds(randperm(length(fg_inds), fg_num));
    
    bg_num = min(rois_per_image - fg_rois_per_image, length(bg_inds));
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num));
    
    bg_weight = 1;
    label_weights = zeros(size(bbox_targets, 1), 1);
    label_weights(fg_inds) = fg_rois_per_image/fg_num;
    label_weights(bg_inds) = bg_weight;
    
    if safeDefault(conf, 'iou_importance_weight') && ~isempty(fg_inds) && ~isempty(overlaps) && any(overlaps>0)
       
        label_weights(fg_inds) = label_weights(fg_inds).*(overlaps(fg_inds))/conf.fg_thresh;
    end
    
    bbox_targets = single(full(bbox_targets(:, 2:end)));
    
    if conf.use_recursive && conf.recursive_boxes && conf.iter > conf.recursive_start
        
        % store original mean for fg regions
        changed_fg = fg_to_fg | fg_to_bg;
        mean_orig = mean(label_weights(changed_fg));
        
        % update soft weight ratios
        label_weights(fg_to_fg) = label_weights(fg_to_fg).*pred(fg_to_fg);
        label_weights(fg_to_bg) = label_weights(fg_to_bg).*(1-pred(fg_to_bg));
        
        % update so fg region has same mean weight
        mean_new = mean(label_weights(changed_fg));
        label_weights(changed_fg) = (label_weights(changed_fg)/mean_new)*mean_orig;
        
        % compute bbox_target foregrounds using fg_inds from above
        % but also adding some regular ones to mean the desired fg_num
        fg_inds_all = find(labels_fg_orig);
        fg_inds_not = setdiff(fg_inds_all, fg_inds);
        
        fg_num = min(fg_rois_per_image, length(fg_inds_all));
        fg_needed = fg_num - length(fg_inds);
        
        fg_inds_not = fg_inds_not(randperm(length(fg_inds_not), fg_needed));
        fg_inds = union(fg_inds, fg_inds_not);
        assert(length(fg_inds)==fg_num);
        
    end
    
    bbox_loss_weights = bbox_targets * 0;
    fg_num = length(find(bbox_targets_orig(:, 1) > 0));
    bbox_loss_weights(find(bbox_targets_orig(:, 1) > 0), :) = fg_rois_per_image / fg_num;

    % ---- below is unused in general, except for a few experiments -----
    
    % find best gts
    
    fg_inds = find(bbox_targets_orig(:, 1) > 0);
    bg_inds = find(bbox_targets_orig(:, 1) < 0);
    
    % best fg
    fg_best = fg_inds;
    bg_best = bg_inds;
    
    ignores = find(bbox_targets_orig(:, 1) ==0);
    
    if length(gt_ignores)>0
        best_fg_inds = [];
        tars = unique(targets);
        
        for tarind=1:length(tars)

            tar = tars(tarind);
            if ~gt_ignores(tarind)
                ols = overlaps.*(targets==tar);
                [vals, inds] = sort(ols, 'descend');
                inds = inds(vals>0);
                %[val, ind] = max(ols);
                num_gts_to_add = min(length(inds), conf.best_gt_num);
                best_fg_inds = [best_fg_inds inds(1:num_gts_to_add)];
            end
        end

       % add all non-best to bg_inds
       bg_best = [setdiff(bg_inds, best_fg_inds); setdiff(fg_inds, best_fg_inds)];
       bg_best = setdiff(bg_inds, ignores);
       fg_best = best_fg_inds;
    end
    
    % --------- fg_thresh2 --------
    if isfield(conf, 'fg_thresh2')
        fg_thresh2 = conf.fg_thresh2;
    else
        fg_thresh2 = 0.7;
    end
    
    if isempty(boxes)
       [bbox_targets_2, ~, ~] = ...
           proposal_compute_targets(conf, boxes, gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale, fg_thresh2);
    else
       [bbox_targets_2, ~, ~] = ...
           proposal_compute_targets(conf, scale_rois(boxes, image_roidb.im_size, im_scale), gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale, fg_thresh2);
    end
    
    fg_inds = find(bbox_targets_2(:, 1) > 0);
    bg_inds = find(bbox_targets_2(:, 1) < 0);
    
    % set foreground labels
    labels2 = zeros(size(bbox_targets_2, 1), 1);
    labels2(fg_inds) = 1;
    
    % select foreground
    fg_num = min(fg_rois_per_image, length(fg_inds));
    fg_inds = fg_inds(randperm(length(fg_inds), fg_num));
    
    bg_num = min(rois_per_image - fg_rois_per_image, length(bg_inds));
    bg_inds = bg_inds(randperm(length(bg_inds), bg_num));
    
    bg_weight = 1;
    label_weights2 = zeros(size(bbox_targets_2, 1), 1);
    label_weights2(fg_inds) = fg_rois_per_image/fg_num;
    label_weights2(bg_inds) = bg_weight;
    
    if safeDefault(conf, 'iou_importance_weight') && ~isempty(fg_inds) && ~isempty(overlaps) && any(overlaps>0)
       
        label_weights2(fg_inds) = label_weights2(fg_inds).*(overlaps(fg_inds))/fg_thresh2;
    end
    
    if safeDefault(conf, 'easy_fg_thresh2')
        if isempty(boxes)
           [tmp_targets, ~, ~] = ...
               proposal_compute_targets(conf, boxes, gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale, conf.easy_fg_thresh2);
        else
           [tmp_targets, ~, ~] = ...
               proposal_compute_targets(conf, scale_rois(boxes, image_roidb.im_size, im_scale), gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale, conf.easy_fg_thresh2);
        end
        tmp_labels = tmp_targets(:, 1) > 0;
        label_weights2(tmp_labels & ~labels2) = 0;
    end
    
    
    % --------- fg_thresh3 --------
    if isfield(conf, 'fg_thresh3')
        
        fg_thresh3 = conf.fg_thresh3;
        
        if isempty(boxes)
           [bbox_targets_3, ~, ~] = ...
               proposal_compute_targets(conf, boxes, gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale, fg_thresh3);
        else
           [bbox_targets_3, ~, ~] = ...
               proposal_compute_targets(conf, scale_rois(boxes, image_roidb.im_size, im_scale), gt_ignores, image_roidb.class,  anchors{1}, image_roidb, im_scale, fg_thresh3);
        end

        fg_inds = find(bbox_targets_3(:, 1) > 0);
        bg_inds = find(bbox_targets_3(:, 1) < 0);

        % set foreground labels
        labels3 = zeros(size(bbox_targets_3, 1), 1);
        labels3(fg_inds) = 1;

        % select foreground
        fg_num = min(fg_rois_per_image, length(fg_inds));
        fg_inds = fg_inds(randperm(length(fg_inds), fg_num));

        bg_num = min(rois_per_image - fg_rois_per_image, length(bg_inds));
        bg_inds = bg_inds(randperm(length(bg_inds), bg_num));

        bg_weight = 1;
        label_weights3 = zeros(size(bbox_targets_3, 1), 1);
        label_weights3(fg_inds) = fg_rois_per_image/fg_num;
        label_weights3(bg_inds) = bg_weight;
        if safeDefault(conf, 'iou_importance_weight') && ~isempty(fg_inds) && ~isempty(overlaps) && any(overlaps>0)
       
            label_weights3(fg_inds) = label_weights3(fg_inds).*(overlaps(fg_inds))/fg_thresh3;
        end
    else
       
        fg_inds = fg_best;
        fg_num = min(fg_rois_per_image, length(fg_inds));
        fg_inds = fg_inds(randperm(length(fg_inds), fg_num));

        bg_inds = bg_best;
        bg_num = min(rois_per_image - fg_rois_per_image, length(bg_inds));
        bg_inds = bg_inds(randperm(length(bg_inds), bg_num));
        labels3 = zeros(size(bbox_targets_orig, 1), 1);

        labels3(fg_inds) = 1;

        bg_weight = 1;
        label_weights3 = zeros(size(bbox_targets_orig, 1), 1);
        label_weights3(fg_inds) = fg_rois_per_image/fg_num;
        label_weights3(bg_inds) = bg_weight;

        overlaps = full(overlaps);

        if safeDefault(conf, 'overwrite_best_with_iou')

            label_weights3  =  label_weights;
            labels3         =  overlaps;

            if isfield(conf, 'iou_power')

                for bind=1:size(boxes,1)

                    maxval = max(overlaps(targets==bind));
                    labels3(targets==bind) = overlaps(targets==bind)/maxval;
                end

                labels3 = labels3.^conf.iou_power;

                %{
                if any(overlaps>=conf.fg_thresh)
                    minval   = min(labels3(overlaps>=conf.fg_thresh));
                    bweights = labels3/minval;
                else
                    bweights = labels3;
                end

                ignore_regions = label_weights==0;

                label_weights3 = bweights;
                label_weights3(overlaps<conf.fg_thresh) = 1; % "bg" 
                label_weights3(ignore_regions)          = 0; % ignore

                bbox_loss_weights(:,1) = bweights;
                bbox_loss_weights(:,2) = bweights;
                bbox_loss_weights(:,3) = bweights;
                bbox_loss_weights(:,4) = bweights;
                %}
            end
        end
        
        
    end
    
end

function scaled_rois = scale_rois(rois, im_size, im_scale)
    im_size_scaled = round(im_size * im_scale);
    scale = (im_size_scaled - 1) ./ (im_size - 1);
    scaled_rois = bsxfun(@times, rois-1, [scale(2), scale(1), scale(2), scale(1)]) + 1;
end

