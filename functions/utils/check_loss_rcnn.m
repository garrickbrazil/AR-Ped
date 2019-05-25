function cur_results = check_loss_rcnn(conf, caffe_solver, cur_results, im, rois, debug_flag)

    % loss values
    for lossind=1:length(conf.loss_layers)
        
        loss_layer = conf.loss_layers{lossind};
        
        if ~isfield(cur_results, loss_layer), cur_results.(loss_layer) = []; end
        
        lossval = caffe_solver.net.blobs(loss_layer).get_data();
        
        if lossval > -1
            
            % artificial boost for bbox regression
            if strcmp(loss_layer, 'loss_bbox'), lossval = lossval*1; end
            
            cur_results.(loss_layer) = [cur_results.(loss_layer) lossval];
        end
        
    end
    
    % accuracy
    labels = caffe_solver.net.blobs('labels').get_data();
    pred = caffe_solver.net.blobs('cls_score_sm2').get_data();
    scores = pred(end,:);
    
    %{
    if isfield(conf, 'contrastive') && conf.contrastive && any(labels(:)>0)
       ids     = caffe_solver.net.blobs('id').get_data(); 
       ols     = caffe_solver.net.blobs('ols').get_data(); 
       targets = caffe_solver.net.blobs('targets').get_data(); 
       
       ids     = ids(:);
       targets = targets(:);
       ols     = ols(:);
       
       ids     = ids(ols>0.5);
       targets = targets(ols>0.5);
       
       [targets, inds] = sort(targets, 'descend');
       ids = ids(inds);
       
    end
    %}
    
    [~, pred] = max(pred, [], 1);
    labels = labels(:); 
    pred = pred(:)-1;
    
    if debug_flag
        top     = [];
        bottom  = [];
        for i=1:length(labels)
            x1 = rois.boxes(i,1);
            y1 = rois.boxes(i,2);
            x2 = rois.boxes(i,3);
            y2 = rois.boxes(i,4);
            w = x2-x1;
            h = y2-y1;

            x1 = x1 - w*conf.padfactor;
            y1 = y1 - h*conf.padfactor;
            w = w + w*conf.padfactor;
            h = h + h*conf.padfactor;

            crop = imcrop(im, [x1 y1 w h]);
            crop = imresize(crop, [800 round(800*0.41)]);
            if rois.ols(i)>0
                target = rois.targets(i);
                ignore = rois.ignores(target);
            else
                target = -1;
                ignore = -1;
            end
            
            rpn_score = exp(rois.feat_scores_fg(i))/(exp(rois.feat_scores_bg(i)) + exp(rois.feat_scores_fg(i)));
            
            c = 6;
            if labels(i)==1
                crop = insertText(crop, [c c], sprintf('Lbl=%d, Ol=%.2f, Ign=%d, Pred=%d, Sco=%.2f, RPN=%.2f',labels(i), rois.ols(i), ignore, pred(i), scores(i), rpn_score), 'FontSize', 10, 'BoxColor', 'green');
            else
                crop = insertText(crop, [c c], sprintf('Lbl=%d, Ol=%.2f, Ign=%d, Pred=%d, Sco=%.2f, RPN=%.2f',labels(i), rois.ols(i), ignore, pred(i), scores(i), rpn_score), 'FontSize', 10, 'BoxColor', 'red');
            end
            
            if pred(i)==1
                crop = addBorder(crop, [0, 200, 0], c, 0);
            else
                crop = addBorder(crop, [200, 0, 0], c, 0);
            end
            if labels(i)==1
                %crop = addBorder(crop, [0, 200, 0], c, c);
            else
                %crop = addBorder(crop, [200, 0, 0], c, c);
            end
            crop = addBorder(crop, [0 0 0], 1,0);
            
            if i<=10
                top = [top crop];
            else
                bottom = [bottom crop];
            end
            
        end
        if size(top, 2)>size(bottom,2)
            bottom = padarray(bottom, [0 abs(size(top,2) - size(bottom,2)) 0], 'symmetric', 'post');
        elseif size(top,2)<size(bottom,2)
            top = padarray(top, [0 abs(size(top,2) - size(bottom,2)) 0], 'symmetric', 'post');
        end
        
        out = [top; bottom];
        imshow(out); drawnow;
        tmp_dir = '/home/gbmsu/Desktop/city_rcnn_vis/';
        mkdir_if_missing(tmp_dir);
        imwrite(out, [tmp_dir rois.image_id '.jpg']);
    end
    
    fg_acc = sum(pred(labels>0)==labels(labels>0))/sum(labels>0);
    bg_acc = sum(pred(labels==0)==labels(labels==0))/sum(labels==0);
    acc    = sum(pred==labels)/length(labels);
    
    if ~isfield(cur_results, 'fg_acc'), cur_results.fg_acc = []; end
    if ~isfield(cur_results, 'bg_acc'), cur_results.bg_acc = []; end
    if ~isfield(cur_results, 'acc'),    cur_results.acc    = []; end
    
    if sum(labels>0)  > 0, cur_results.fg_acc = [cur_results.fg_acc fg_acc]; end
    if sum(labels==0) > 0, cur_results.bg_acc = [cur_results.bg_acc bg_acc]; end
    cur_results.acc    = [cur_results.acc acc];
    
end

function im = addBorder(im, colour, c, s)

    c = uint32(c);
    s = uint32(s);
    colour = uint8(colour);
    
    im(1+s:c+s,1+s:end-s,1)       = colour(1); % top R
    im(1+s:end-s,1+s:c+s,1)       = colour(1); % left R
    im(1+s:end-s,end-c-s:end-s,1) = colour(1); % right R
    im(end-c-s:end-s,1+s:end-s,1) = colour(1); % bottom R

    im(1+s:c+s,1+s:end-s,2)       = colour(2); % top G
    im(1+s:end-s,1+s:c+s,2)       = colour(2); % left G
    im(1+s:end-s,end-c-s:end-s,2) = colour(2); % right G
    im(end-c-s:end-s,1+s:end-s,2) = colour(2); % bottom G

    im(1+s:c+s,1+s:end-s,3)       = colour(3); % top B
    im(1+s:end-s,1+s:c+s,3)       = colour(3); % left B
    im(1+s:end-s,end-c-s:end-s,3) = colour(3); % right B
    im(end-c-s:end-s,1+s:end-s,3) = colour(3); % bottom B

end