function auto_data = get_radial(im, conf, rcnn_conf, boxes, rpn_net)

    auto_raw_data = [];
    proposal_num = size(boxes,1);
    
    for layerind=1:length(rcnn_conf.radial_layers)
        data = rpn_net.blobs(rcnn_conf.radial_layers{layerind}).get_data();
        data = permute(data, [2, 1, 3, 4]);

        if isempty(auto_raw_data)
            auto_raw_data = data/length(rcnn_conf.radial_layers); 
        else
            auto_raw_data = auto_raw_data + data/length(rcnn_conf.radial_layers);
        end
    end

    auto_data = zeros(rcnn_conf.weak_seg_crop(1), rcnn_conf.weak_seg_crop(2), size(auto_raw_data,3), proposal_num, 'single');
    sf = conf.test_scale/(size(im,1)*conf.feat_stride);

    bigpad_h = round(size(auto_raw_data,1)*rcnn_conf.padfactor*2);
    bigpad_w = round(size(auto_raw_data,2)*rcnn_conf.padfactor*2);

    auto_raw_data = padarray(auto_raw_data, [bigpad_h bigpad_w]);

    for bind=1:size(boxes,1)

        box = boxes(bind,:)*sf;

        x1 = box(1);
        y1 = box(2);
        x2 = box(3);
        y2 = box(4);

        w  = (x2-x1) + 1;
        h  = (y2-y1) + 1;
        padw = w * rcnn_conf.padfactor/2;
        padh = h * rcnn_conf.padfactor/2;

        x1 = max(round(x1-padw + bigpad_w),1);
        x2 = max(round(x2+padw + bigpad_w),1);
        y1 = max(round(y1-padh + bigpad_h),1);
        y2 = max(round(y2+padh + bigpad_h),1);

        crop_data = auto_raw_data(y1:y2,x1:x2,:);
        crop_resize = imresize(crop_data, [rcnn_conf.weak_seg_crop(1), rcnn_conf.weak_seg_crop(2)]);
        auto_data(:,:,:,bind) = crop_resize;
    end

end