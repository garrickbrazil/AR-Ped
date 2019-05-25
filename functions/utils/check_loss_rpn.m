function cur_results = check_loss_rpn(conf, caffe_solver, cur_results)

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
    
    if conf.split_anchors
        lbl_list = {};
        cls_list = {};
        
        for s=conf.anchor_sets_stride
            lbl_list{end+1} = sprintf('labels_reshape_s%d', s);
            cls_list{end+1} = sprintf('proposal_cls_score_reshape_s%d', s);
        end
    else
        lbl_list = {'labels_reshape'};
        cls_list = {'proposal_cls_score_reshape'};
    end
    
    labels = [];
    pred   = [];
    
    for sind=1:length(lbl_list)
    
        % accuracy
        labels_tmp = caffe_solver.net.blobs(lbl_list{sind}).get_data();
        
        try
            pred_tmp = caffe_solver.net.blobs(cls_list{sind}).get_data();
        catch
            pred_tmp = caffe_solver.net.blobs('proposal_cls_score_reshape1').get_data();
        end

        [~, pred_tmp] = max(pred_tmp, [], 3);
        labels_tmp = labels_tmp(:); 
        pred_tmp = pred_tmp(:)-1;
        
        labels = [labels; labels_tmp(:)];
        pred   = [pred; pred_tmp(:)];
    end
    
    acc    = sum(pred==labels)/length(labels);
    
    fg_acc = pred(labels>0)==labels(labels>0);
    bg_acc = pred(labels==0)==labels(labels==0);
    
    if ~isfield(cur_results, 'fg_acc'), cur_results.fg_acc = []; end
    if ~isfield(cur_results, 'bg_acc'), cur_results.bg_acc = []; end
    if ~isfield(cur_results, 'acc'),    cur_results.acc    = []; end
    
    if sum(labels>0)  > 0, cur_results.fg_acc = logical([cur_results.fg_acc fg_acc']); end
    if sum(labels==0) > 0, cur_results.bg_acc = logical([cur_results.bg_acc bg_acc']); end
    cur_results.acc    = [cur_results.acc acc];
    
end
