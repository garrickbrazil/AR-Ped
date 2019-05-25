function [mr_fused, mr_rcnn] = evaluate_results_rcnn(rcnn_conf, net, test_rois, db, results_dir_fused, results_dir_rcnn, mode, rpn_conf, rpn_net)


    if (exist(results_dir_fused, 'dir')), rmdir(results_dir_fused, 's'); end
    if (exist(results_dir_rcnn, 'dir')),  rmdir(results_dir_rcnn, 's'); end

    for testind=1:length(test_rois)

        rois = test_rois{testind};

        reg = regexp(rois.image_id, '(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)', 'tokens');
        setname = reg{1}{1};
        vname = reg{1}{2};
        iname = reg{1}{3};
        inum = str2num(iname) + 1;

        mkdir_if_missing([results_dir_fused '/' setname]);
        mkdir_if_missing([results_dir_rcnn  '/' setname]);

        fid  = fopen([results_dir_fused '/' setname '/' vname '.txt'], 'a');
        fid2 = fopen([results_dir_rcnn  '/' setname '/' vname '.txt'], 'a');

        if isfield(rcnn_conf, 'share_layer')
            [net_inputs, ~, valid] = get_rcnn_batch(rcnn_conf, rois, mode, rpn_conf, rpn_net);
        else
            [net_inputs, ~, valid] = get_rcnn_batch(rcnn_conf, rois, mode);
        end
        
        if ~any(valid)
            fclose(fid);
            fclose(fid2);
            continue;
        end
        
        net = reshape_input_data(net, net_inputs);
        net.forward(net_inputs);

        cls_scores_fused = net.blobs('cls_score_sm').get_data();
        cls_scores_fused = cls_scores_fused(end,:);
        cls_scores_fused = cls_scores_fused(:);

        cls_scores_rcnn = net.blobs('cls_score2_sm').get_data();
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
        bbox_reg = net.blobs('bbox_reg').get_data();
        
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

        % score 2 (rcnn only)
        aboxes = [boxes, cls_scores_rcnn];

        if safeDefault(rcnn_conf, 'has_bbox_reg')
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

            fprintf(fid2, '%d,%.3f,%.3f,%.3f,%.3f,%.6f\n', [inum x1 y1 w h score]);

        end

        fclose(fid);
        fclose(fid2);

    end

    mr_fused = evaluate_result_dir({results_dir_fused}, db, rcnn_conf.test_min_h);
    mr_rcnn  = evaluate_result_dir({results_dir_rcnn}, db, rcnn_conf.test_min_h);

end