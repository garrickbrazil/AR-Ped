function test_rcnn(rpn_prototxt, rpn_weights, rpn_conf, anchors, bbox_means, bbox_stds, rcnn_prototxt, rcnn_weights, rcnn_conf, gpu_id)

    if ~exist('gpu_id',      'var'),  gpu_id       =  1;               end
    
    rpn_conf.gpu_id     = gpu_id;
    rpn_conf.anchors    = anchors;
    rpn_conf.bbox_means = bbox_means;
    rpn_conf.bbox_stds  = bbox_stds;
    
    warning('off', 'MATLAB:class:DestructorError');
    
    fprintf('Processing test.. ');

    reset_caffe(rpn_conf);
    
    test_dir     = [pwd '/datasets/' rpn_conf.dataset_test  '/test'];
    results_dir = [pwd '/.tmpresults'];
    
    if (exist(results_dir, 'dir')), rmdir(results_dir, 's'); end

    rpn_net = caffe.Net(rpn_prototxt, 'test');
    rpn_net.copy_from([rpn_weights]);
    
    rcnn_net = caffe.Net(rcnn_prototxt, 'test');
    rcnn_net.copy_from([rcnn_weights]);
    
    imlist = dir([test_dir '/images/*.jpg']);
    
    rcnn_conf.test_dir = test_dir;
    
    start_test = tic;
    
    for imind=1:length(imlist)
        
        %imobj = imlist{imind};
        im = imread([test_dir '/images/' imlist(imind).name]);
        
        [boxes, scores, feat_scores_bg, feat_scores_fg] = proposal_im_detect(rpn_conf, rpn_net, im);
        
        % filter rpn
        proposal_num = rcnn_conf.test_batch_size;
        [aboxes, inds] = nms_filter([boxes, scores], rpn_conf.nms_per_nms_topN, rpn_conf.nms_overlap_thres, proposal_num, 1);
        
        boxes = aboxes(:, 1:4);
        scores = aboxes(:, 5);
        
        feat_scores_bg = feat_scores_bg(inds,:);
        feat_scores_fg = feat_scores_fg(inds,:);
        
        feat_scores_bg = feat_scores_bg(1:min(length(aboxes), proposal_num), :);
        feat_scores_fg = feat_scores_fg(1:min(length(aboxes), proposal_num), :);
        
        rois = {};
        
        [~,image_id] = fileparts(imlist(imind).name);
        
        reg = regexp(image_id, '(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)', 'tokens');
        setname = reg{1}{1};
        vname = reg{1}{2};
        iname = reg{1}{3};
        inum = str2num(iname) + 1;

        mkdir_if_missing([results_dir '/' setname]);

        fid  = fopen([results_dir '/' setname '/' vname '.txt'], 'a');
        
        rois.image_id = image_id;
        rois.boxes = single(boxes);
        rois.scores = single(scores);
        rois.feat_scores_bg = single(feat_scores_bg);
        rois.feat_scores_fg = single(feat_scores_fg);
        
        [net_inputs, ~, valid] = get_rcnn_batch(rcnn_conf, rois, 'test');
        
        if ~any(valid)
            fclose(fid);
            continue;
        end
        
        rcnn_net = reshape_input_data(rcnn_net, net_inputs);
        rcnn_net.forward(net_inputs);
        
        cls_scores_fused = rcnn_net.blobs('cls_score_sm').get_data();
        cls_scores_fused = cls_scores_fused(end,:);
        cls_scores_fused = cls_scores_fused(:);

        cls_scores_rcnn = rcnn_net.blobs('cls_score2_sm').get_data();
        cls_scores_rcnn = cls_scores_rcnn(end,:);
        cls_scores_rcnn = cls_scores_rcnn(:);

        boxes = rois.boxes(1:rcnn_conf.test_batch_size,:);
        boxes = boxes(valid,:);
        
        if safeDefault(rcnn_conf, 'soft_suppress')
            boxols = boxoverlap(boxes, boxes);
            boxols = boxols .* single(boxols~=1);

            soft = false;
            olthresh = 0.4;
            vari     = 0.3;

            boxeswithol = boxols>olthresh;

            suppress = triu(boxeswithol);
            cls_scores_fused_tmp = cls_scores_fused;

            [r, c] = find(suppress);
            for i=1:size(r,1)

                score1 = cls_scores_fused(r(i));
                score2 = cls_scores_fused(c(i));
                olscore = boxols(r(i),c(i));
                if soft
                    penalty = exp((-(olscore)^2)/vari);
                else
                    penalty = 1 - olscore;
                end

                if score1>score2
                   cls_scores_fused_tmp(c(i)) = cls_scores_fused_tmp(c(i))*penalty;
                else
                   cls_scores_fused_tmp(r(i)) = cls_scores_fused_tmp(r(i))*penalty;
                end
            end

            cls_scores_fused = cls_scores_fused_tmp;
        end
        
        % score 1 (fused)
        aboxes = [boxes, cls_scores_fused];
        if safeDefault(rcnn_conf, 'has_bbox_reg')
        bbox_reg = rcnn_net.blobs('bbox_reg').get_data();
        
        % apply bbox reg
         for scoreind=1:size(aboxes,1)

            x1 = aboxes(scoreind, 1);
            y1 = aboxes(scoreind, 2);
            x2 = aboxes(scoreind, 3);
            y2 = aboxes(scoreind, 4);

            cx = (x1 + x2)/2;
            cy = (y1 + y2)/2;

            w = x2 - x1+1;
            h = y2 - y1+1;

            dx = bbox_reg(1,scoreind);
            dy = bbox_reg(2,scoreind);
            dw = bbox_reg(3,scoreind);
            dh = bbox_reg(4,scoreind);

            % apply to box
            new_cx = cx + dx*w;
            new_cy = cy + dy*h;
            new_w  = w/exp(dw);
            new_h  = h/exp(dh);
            new_x1 = (2*new_cx - new_w + 1)/2;
            new_y1 = (2*new_cy - new_h + 1)/2;
            new_x2 = new_x1 + new_w - 1;
            new_y2 = new_y1 + new_h - 1;
            aboxes(scoreind,1:4) = [new_x1, new_y1, new_x2, new_y2];
         end

        [aboxes, ~] = nms_filter(aboxes, size(aboxes,1), 0.5, size(aboxes,1) , true);
        end
        
        for scoreind=1:size(aboxes,1)

            x1 = aboxes(scoreind, 1);
            y1 = aboxes(scoreind, 2);
            x2 = aboxes(scoreind, 3);
            y2 = aboxes(scoreind, 4);
            score = aboxes(scoreind, 5);

            w = x2 - x1;
            h = y2 - y1;

            fprintf(fid, '%d,%.3f,%.3f,%.3f,%.3f,%.6f\n', [inum x1 y1 w h score]);

        end
        
        if mod(imind, 20)==0
            [timeleft, dt] = compute_eta(toc(start_test), imind, length(imlist));
            fprintf('%d/%d, dt: %0.4f, eta: %s\n', imind, length(imlist), dt, timeleft);
        end
        
        fclose(fid);
        
    end
    
    mr = evaluate_result_dir({results_dir}, rcnn_conf.test_db, rcnn_conf.test_min_h);
    
    fprintf('MR=%.4f\n', mr);
    
    reset_caffe(rpn_conf);
    
    %if (exist(results_dir, 'dir')), rmdir(results_dir, 's'); end

end
