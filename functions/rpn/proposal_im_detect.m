function [pred_boxes_all, scores_all, feat_scores_bg_all, feat_scores_fg_all, cls_all, scores_unfiltered, caffe_net] = proposal_im_detect(conf, caffe_net, im, extras)
% [pred_boxes, scores, feat_scores_bg, feat_scores_fg] = proposal_im_detect(conf, caffe_net, im)
% --------------------------------------------------------
% RPN_BF
% Copyright (c) 2016, Liliang Zhang
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    

    if ~isfield(conf, 'test_scale')
        conf.test_scale = conf.scales;
    end
    
    im = single(im);
    [im_blob, im_scales] = prep_im_for_blob(im, conf.image_means, conf.test_scale, conf.max_size);
    im_size = size(im);
    scaled_im_size = round(im_size * im_scales);
    
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg    
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);
    
    net_inputs = {im_blob};

    if safeDefault(conf, 'has_nms_lstm')
        seq1  = zeros(1, 1);
        seq2  = ones(1, 1);
        seq3  = ones(1, 1);
        hin   = zeros(60, 45, 512, 1, 1);
        cout  = zeros(60, 45, 512, 1, 1);
        %{
        net_inputs{end+1} = seq1;
        net_inputs{end+1} = seq2;
        net_inputs{end+1} = seq3;
        net_inputs{end+1} = hin;
        net_inputs{end+1} = cout;
        %}
    end
    
    if safeDefault(conf, 'has_history')
        net_inputs = {net_inputs{:} extras{:}};
        caffe_net = reshape_input_data(caffe_net, net_inputs);
        caffe_net.forward(net_inputs);
        
    elseif safeDefault(conf, 'has_lstm')
        
        extras.net = reshape_input_data(extras.net, net_inputs);
        extras.net.forward(net_inputs);
        
        lstm_w      = size(extras.net.blobs(conf.lstm_layer).get_data(),1);
        lstm_h      = size(extras.net.blobs(conf.lstm_layer).get_data(),2);
        feat_dim    = size(extras.net.blobs(conf.lstm_layer).get_data(),3);
        data_inputs = zeros(lstm_w, lstm_h, feat_dim, 1, 1);
        seq_inputs  = ones(1, 1);
        
        data_inputs(:,:,:, 1, 1) = extras.net.blobs(conf.lstm_layer).get_data();
        net_inputs = {data_inputs, seq_inputs};
        
        net_inputs{end+1} = extras.h_cls;
        net_inputs{end+1} = extras.c_cls;
        net_inputs{end+1} = extras.h_bbox;
        net_inputs{end+1} = extras.c_bbox;
        
        caffe_net.forward(net_inputs);
        
        bbox_pred_layer = 'lstm_bbox_reshape';

    elseif safeDefault(conf, 'has_nms_lstm')
        caffe_net.forward(net_inputs);
    else    
        caffe_net = reshape_input_data(caffe_net, net_inputs);
        caffe_net.forward(net_inputs);
    end
    
    if safeDefault(conf, 'split_anchors')
        bbox_list = {};
        cls_list  = {};
        feat_list = {};
        
        for s=conf.anchor_sets_stride
            bbox_list{end+1} = sprintf('proposal_bbox_pred_s%d', s);
            cls_list{end+1}  = sprintf('proposal_cls_prob_s%d', s);
            feat_list{end+1} = sprintf('proposal_cls_score_reshape_s%d', s);
        end
    else
        bbox_list = {'proposal_bbox_pred'};
        cls_list  = {'proposal_cls_prob'};
        feat_list = {'proposal_cls_score_reshape'};
    end
    
    if safeDefault(conf, 'custom_feat_cls_layer')
       feat_list = {conf.custom_feat_cls_layer};
    end
    if safeDefault(conf, 'custom_cls_layer')
       cls_list = {conf.custom_cls_layer};
    end
    if safeDefault(conf, 'custom_bbox_layer')
       bbox_list = {conf.custom_bbox_layer};
    end
    
    pred_boxes_all = [];
    scores_all = [];
    feat_scores_bg_all = [];
    feat_scores_fg_all = [];
    cls_all = []; 
    
    for sind=1:length(bbox_list)

        bbox_pred_layer   = bbox_list{sind};
        cls_pred_layer    = cls_list{sind};
        feature_cls_layer = feat_list{sind};
        
        if safeDefault(conf, 'split_anchors')
            conf.anchor_scales = conf.anchor_sets(sind,:);
            conf.anchors = proposal_generate_anchors(conf, true);
            conf.feat_stride = conf.anchor_sets_stride(sind);
        end

        % Apply bounding-box regression deltas
        
        try
            box_deltas = caffe_net.blobs(bbox_pred_layer).get_data();
        catch
            box_deltas = caffe_net.blobs('proposal_bbox_pred1').get_data();
            bbox_pred_layer = 'proposal_bbox_pred1';
        end
        
        featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];

        % permute from [width, height, channel] to [channel, height, width], where channel is the fastest dimension
        box_deltas = permute(box_deltas, [3, 2, 1]);
        box_deltas = reshape(box_deltas, 4, [])';

        if isfield(conf, 'manual_bbox_means') && conf.manual_bbox_means
            box_deltas(:,1) = box_deltas(:,1)*conf.bbox_stds(1) + conf.bbox_means(1);
            box_deltas(:,2) = box_deltas(:,2)*conf.bbox_stds(2) + conf.bbox_means(2);
            box_deltas(:,3) = box_deltas(:,3)*conf.bbox_stds(3) + conf.bbox_means(3);
            box_deltas(:,4) = box_deltas(:,4)*conf.bbox_stds(4) + conf.bbox_means(4);
        end

        anchors = proposal_locate_anchors(conf, size(im), conf.test_scale, featuremap_size);
        pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas);

        % scale back
        pred_boxes = bsxfun(@times, pred_boxes - 1, ...
            ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
        pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));

        % use softmax estimated probabilities
        scores = caffe_net.blobs(cls_pred_layer).get_data();
        %scores(:,:,1) = []; % remove background class
        scores = scores(:, :, end);
        [scores, cls] = max(scores,[], 3);
        scores = reshape(scores, size(caffe_net.blobs(bbox_pred_layer).get_data(), 1), size(caffe_net.blobs(bbox_pred_layer).get_data(), 2), []);
        cls    = reshape(cls, size(caffe_net.blobs(bbox_pred_layer).get_data(), 1), size(caffe_net.blobs(bbox_pred_layer).get_data(), 2), []);

        % store features
        try
            feat_scores = caffe_net.blobs(feature_cls_layer).get_data();
        catch
            feat_scores = caffe_net.blobs('proposal_cls_score_reshape3').get_data();
        end
        
        feat_scores_bg = feat_scores(:, :, 1);
        feat_scores_fg = feat_scores(:, :, 2);

        feat_scores_fg = reshape(feat_scores_fg, size(caffe_net.blobs(bbox_pred_layer).get_data(), 1), size(caffe_net.blobs(bbox_pred_layer).get_data(), 2), []);
        feat_scores_fg = permute(feat_scores_fg, [3, 2, 1]);
        feat_scores_fg = feat_scores_fg(:);
        feat_scores_bg = reshape(feat_scores_bg, size(caffe_net.blobs(bbox_pred_layer).get_data(), 1), size(caffe_net.blobs(bbox_pred_layer).get_data(), 2), []);
        feat_scores_bg = permute(feat_scores_bg, [3, 2, 1]);
        feat_scores_bg = feat_scores_bg(:);

        % permute from [width, height, channel] to [channel, height, width], where channel is the
            % fastest dimension
        scores = permute(scores, [3, 2, 1]);
        cls    = permute(cls, [3, 2, 1]);
        scores = scores(:);
        cls = cls(:);
        
        pred_boxes_all = [pred_boxes_all; pred_boxes];
        scores_all = [scores_all; scores];
        feat_scores_bg_all = [feat_scores_bg_all; feat_scores_bg];
        feat_scores_fg_all = [feat_scores_fg_all; feat_scores_fg];
        cls_all = [cls_all; cls]; 
    
    end
    
    scores_unfiltered = scores_all; 
    
    % drop too small boxes
    [pred_boxes_all, scores_all, valid_ind] = filter_boxes(conf.test_min_box_size, conf.test_min_box_height, pred_boxes_all, scores_all);
    
    % sort
    [scores_all, scores_ind] = sort(scores_all, 'descend');
    pred_boxes_all = pred_boxes_all(scores_ind, :);
    
    feat_scores_fg_all = feat_scores_fg_all(valid_ind, :);
    feat_scores_fg_all = feat_scores_fg_all(scores_ind, :);
    feat_scores_bg_all = feat_scores_bg_all(valid_ind, :);
    feat_scores_bg_all = feat_scores_bg_all(scores_ind, :);
    
    cls_all = cls_all(valid_ind);
    cls_all = cls_all(scores_ind);
    
    if safeDefault(conf, 'split_anchors')
        conf.anchor_scales = conf.orig_anchor_scales;
        conf.anchors       = conf.orig_anchors;
        conf.feat_stride   = conf.orig_feat_stride;
    end
    
end

function [boxes, scores, valid_ind] = filter_boxes(min_box_size, min_box_height, boxes, scores)
    widths = boxes(:, 3) - boxes(:, 1) + 1;
    heights = boxes(:, 4) - boxes(:, 2) + 1;
    
    valid_ind = widths >= min_box_size & heights >= min_box_size & heights >= min_box_height;
    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
end
    
function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end
    
