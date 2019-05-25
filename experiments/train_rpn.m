function out_jargin = train_rpn(config_name, gpu_id, solverstate)

    % ================================================
    % basic configuration
    % ================================================ 
    
    debug_im = false;
    
    % defaults
    if ~exist('config_name', 'var'),  error('Please provide config');  end
    if ~exist('gpu_id',      'var'),  gpu_id       =  1;               end
    if ~exist('solverstate', 'var'),  solverstate  =  '';              end
    
    rpn_conf = Config.rpn.(config_name);
    
    rpn_conf.stage        =  'rpn';
    rpn_conf.config_name  =  config_name;
    
    % anchor splitting
    if ~isfield(rpn_conf, 'split_anchors')
        rpn_conf.split_anchors = false;
    elseif rpn_conf.split_anchors
        
        % store orig
        rpn_conf.orig_anchor_scales  = rpn_conf.anchor_scales;
        rpn_conf.orig_feat_stride    = rpn_conf.feat_stride;

        anchor_sets_count = length(rpn_conf.anchor_sets_stride);
        anchors_per_set   = length(rpn_conf.anchor_scales)/anchor_sets_count;
        
        assert(mod(anchors_per_set, 1)==0);
        
        asets = zeros(anchor_sets_count, anchors_per_set, 'single');
        
        % assume coarse to fine order of sets
        for asetind=1:anchor_sets_count
            strind = ((asetind-1)*anchors_per_set+1);
            endind = strind+anchors_per_set-1;
            asets(asetind,:) = rpn_conf.orig_anchor_scales(strind:endind);
        end
        
        rpn_conf.anchor_sets = asets;
    end
    
    % cannot trust manual_bbox_means!!
    % gb changed for bug fix
    rpn_conf.manual_bbox_means = true;
    
    % store config
    output_dir = [pwd '/output/' rpn_conf.stage '/' rpn_conf.config_name];
    mkdir_if_missing(output_dir);
    save([output_dir '/rpn_conf.mat'], 'rpn_conf');
    
    % extra misc
    rpn_conf.show_plot    = ~(usejava('jvm') && ~feature('ShowFigureWindows')) && 1;
    rpn_conf.gpu_id       = gpu_id;
    rpn_conf.solverstate  = solverstate;
    
    % extra paths
    rpn_conf.base_dir     = pwd;
    rpn_conf.output_dir   = output_dir;
    rpn_conf.model_dir    = [rpn_conf.base_dir '/models/'     rpn_conf.stage '/' rpn_conf.model];
    rpn_conf.init_weights = [rpn_conf.base_dir '/pretrained/' rpn_conf.pretrained];
    rpn_conf.train_dir    = [rpn_conf.base_dir '/datasets/'   rpn_conf.dataset_train '/train'];
    rpn_conf.test_dir     = [rpn_conf.base_dir '/datasets/'   rpn_conf.dataset_test  '/test'];
    rpn_conf.val_dir      = [rpn_conf.base_dir '/datasets/'   rpn_conf.dataset_val   '/val'];
    rpn_conf.weights_dir  = [rpn_conf.output_dir '/weights'];
    rpn_conf.solver_path  = [rpn_conf.output_dir '/solver.prototxt'];
    rpn_conf.train_path   = [rpn_conf.output_dir '/train.prototxt'];
    rpn_conf.test_path    = [rpn_conf.output_dir '/test.prototxt'];
    rpn_conf.cache_dir    = [rpn_conf.base_dir   '/datasets/cache'];
    rpn_conf.log_dir      = [rpn_conf.output_dir '/log'];
    
    % ================================================
    % setup
    % ================================================ 

    mkdir_if_missing(rpn_conf.weights_dir);
    mkdir_if_missing(rpn_conf.log_dir);
    
    copyfile([rpn_conf.model_dir '/train.prototxt'], rpn_conf.train_path);
    copyfile([rpn_conf.model_dir '/test.prototxt' ], rpn_conf.test_path);
    
    % imdb and roidb
    imdb_train    = imdb_generate(['datasets/' rpn_conf.dataset_train], 'train', false, rpn_conf.cache_dir, rpn_conf.dataset_train, rpn_conf);
    roidb_train   = roidb_generate(imdb_train, false, rpn_conf.cache_dir, rpn_conf.dataset_train, rpn_conf.min_gt_height, rpn_conf);
    
    % anchors
    rpn_conf.anchors = proposal_generate_anchors(rpn_conf);
    
    if rpn_conf.split_anchors
        rpn_conf.orig_anchors = rpn_conf.anchors;
    end
    
    % misc
    write_solver(rpn_conf);
    reset_caffe(rpn_conf);
    rng(rpn_conf.mat_rng_seed);
    warning('off', 'MATLAB:class:DestructorError'); 
    warning('off', 'Images:initSize:adjustingMag');
    warning('off');
    
    % solver
    caffe_solver = caffe.Solver(rpn_conf.solver_path);
    
    if isempty(strfind(rpn_conf.init_weights, '.solverstate'))
        caffe_solver.net.copy_from(rpn_conf.init_weights);
    else
        caffe_solver.restore(rpn_conf.init_weights);
        tmp_net = caffe.Net(rpn_conf.train_path, 'train');
        net_surgery_copy(caffe_solver.net, tmp_net);
        caffe_solver = caffe.Solver(rpn_conf.solver_path);
        net_surgery_copy(tmp_net, caffe_solver.net);
    end
    
    if isfield(rpn_conf, 'pretrained2')
       caffe_solver.net.copy_from([rpn_conf.base_dir '/pretrained/' rpn_conf.pretrained2]);
    end
    
    if length(rpn_conf.solverstate)
        caffe_solver.restore([rpn_conf.output_dir '/weights/' rpn_conf.solverstate '.solverstate']);
    end
    
    % ================================================
    % precompute regressions for all images
    % ================================================ 
    
    rpn_conf.scales = rpn_conf.test_scale;
    
    roidb_train_cache_file  =  [rpn_conf.output_dir '/image_roidb_train.mat'];
    bbox_means_cache_file   =  [rpn_conf.output_dir '/bbox_means.mat'];
    bbox_stds_cache_file    =  [rpn_conf.output_dir '/bbox_stds.mat'];
    
    % preload regression targets
    if exist(roidb_train_cache_file, 'file')==2 && exist(bbox_means_cache_file, 'file')==2 && exist(bbox_stds_cache_file, 'file')==2
        fprintf('Preloading regression targets..');
        load(roidb_train_cache_file);
        load(bbox_means_cache_file);
        load(bbox_stds_cache_file);
        fprintf('Done.\n');
    
    % compute regression targets
    else
        fprintf('Preparing regression targets..');
        [image_roidb_train, bbox_means, bbox_stds] = proposal_prepare_image_roidb(rpn_conf, imdb_train, roidb_train);
        
        if safeDefault(rpn_conf, 'ignore_bbox_means')
           bbox_means = bbox_means*0;
           bbox_stds  = bbox_stds*0 + 1;
        end
        save(roidb_train_cache_file, 'image_roidb_train', '-v7.3');
        save(bbox_means_cache_file,  'bbox_means', '-v7.3');
        save(bbox_stds_cache_file,   'bbox_stds', '-v7.3');
        fprintf('Done.\n');
    end
    
    rpn_conf.bbox_means = bbox_means;
    rpn_conf.bbox_stds  = bbox_stds;
        
    % ================================================
    % train
    % ================================================  
    
    % training
    batch = [];
    all_results = {};
    cur_results = {};
    
    rpn_conf.loss_layers = find_loss_layers(caffe_solver);
    rpn_conf.iter        = caffe_solver.iter();
    
    out_jargin.final_model_path  =  [rpn_conf.output_dir '/final_RPN.caffemodel'];
    out_jargin.output_dir        =  rpn_conf.output_dir;
    
    % already trained?
    if exist(out_jargin.final_model_path, 'file')
        rpn_conf.iter = rpn_conf.max_iter+1;
        fprintf('Final model already exists.\n');
        snapped_file = out_jargin.final_model_path;
    else 
       
        close all; clc; tic;
            
        % log
        curtime = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
        diary([rpn_conf.log_dir '/train_' curtime]);
        caffe.init_log([rpn_conf.log_dir '/caffe_' curtime]);

        disp('conf:'); disp(rpn_conf);
        print_weights(caffe_solver.net);
        
    end
    
    % LSTM
    if safeDefault(rpn_conf, 'has_lstm')
        rpn_conf.test_plain_rpn_path    = [rpn_conf.model_dir '/test_rpn.prototxt'];
        rpn_test_net = caffe.Net(rpn_conf.test_plain_rpn_path, 'test');
        rpn_test_net.copy_from(rpn_conf.init_weights);
    end
    
    rpn_conf.has_nms_lstm = safeDefault(rpn_conf, 'has_nms_lstm');
    
    if rpn_conf.has_nms_lstm
        seq1  = zeros(1, 1);
        seq2  = ones(1, 1);
        seq3  = ones(1, 1);
        hin   = zeros(60, 45, 512, 1, 1);
        cout  = zeros(60, 45, 512, 1, 1);
        
        nms_lstm_inputs = {};
    end
    
    start_iter = rpn_conf.iter;
    
    % recursive
    if rpn_conf.use_recursive
        if rpn_conf.recursive_update==0
            rpn_test_net = caffe_solver.net;
        else
            rpn_test_net = caffe.Net([rpn_conf.model_dir '/test_recursive.prototxt'], 'test');
            net_surgery_copy(caffe_solver.net, rpn_test_net);
        end
    end
    
    if ~isfield(rpn_conf, 'im_batch_size')
        rpn_conf.im_batch_size = 1;
    end
    
    while rpn_conf.iter <= rpn_conf.max_iter
       
        sampleinds = [];
        for bind=1:rpn_conf.im_batch_size
            
            if isempty(batch) && safeDefault(rpn_conf, 'controlled_order')
                rng(rpn_conf.mat_rng_seed);
            end
            
            [batch, sampleinds(end+1)] = proposal_generate_batch(batch, image_roidb_train, 1, rpn_conf.fg_image_ratio);
        end
            
        rpn_conf.scales = datasample(rpn_conf.train_scales, 1, 'Weights', rpn_conf.scales_prob);
        
        rpn_conf.add_noise = rand()<rpn_conf.noise_prob;
        rpn_conf.add_blur  = rand()<rpn_conf.blur_prob;
        rpn_conf.mirror    = rand()<rpn_conf.mirror_prob;
        
        % recursive
        if rpn_conf.use_recursive && rpn_conf.recursive_update~=0 && mod(rpn_conf.iter, rpn_conf.recursive_update)==0
            net_surgery_copy(caffe_solver.net, rpn_test_net);
        end
        
        % LSTM
        if safeDefault(rpn_conf, 'has_lstm')
            
            reg = regexp(image_roidb_train(sampleinds).image_id, '(set\d\d_V\d\d\d)_I\d\d\d\d\d', 'tokens');
            seqid = reg{1}{1};
            
            lstm_inds = sampleinds;
            sampleinds_prev = sampleinds;
            sampleinds_aftr = sampleinds;
            
            while length(lstm_inds)<rpn_conf.lstm_t
                sampleinds_prev = sampleinds_prev-rpn_conf.step;
                sampleinds_aftr = sampleinds_aftr+rpn_conf.step;
                
                % prepend
                if sampleinds_prev > 0 && length(lstm_inds)<rpn_conf.lstm_t
                    reg_tmp = regexp(image_roidb_train(sampleinds_prev).image_id, '(set\d\d_V\d\d\d)_I\d\d\d\d\d', 'tokens');
                    seqid_tmp = reg_tmp{1}{1};
                    if strcmp(seqid, seqid_tmp)
                        lstm_inds = [sampleinds_prev; lstm_inds];
                    end
                end
                
                %append
                if sampleinds_aftr <= length(image_roidb_train) && length(lstm_inds)<rpn_conf.lstm_t
                    reg_tmp = regexp(image_roidb_train(sampleinds_aftr).image_id, '(set\d\d_V\d\d\d)_I\d\d\d\d\d', 'tokens');
                    seqid_tmp = reg_tmp{1}{1};
                    if strcmp(seqid, seqid_tmp)
                        lstm_inds = [lstm_inds; sampleinds_aftr;];
                    end
                end
            end
            
            net_inputs = proposal_generate_minibatch(rpn_conf, image_roidb_train(sampleinds));
            rpn_test_net = reshape_input_data(rpn_test_net, net_inputs(1));
            
            sm_w       = size(net_inputs{2},1);
            sm_h       = size(net_inputs{2},2);
            lstm_w     = size(rpn_test_net.blobs(rpn_conf.lstm_layer).get_data(),1);
            lstm_h     = size(rpn_test_net.blobs(rpn_conf.lstm_layer).get_data(),2);
            feat_dim   = size(rpn_test_net.blobs(rpn_conf.lstm_layer).get_data(),3);
            anchor_num = size(rpn_conf.anchors,1);
            
            data_inputs     = zeros(lstm_w, lstm_h, feat_dim, 1, rpn_conf.lstm_t);
            seq_inputs      = zeros(1, rpn_conf.lstm_t);
            labels_inputs   = zeros(sm_w, sm_h, anchor_num, 1, rpn_conf.lstm_t);
            labels_w_inputs = zeros(sm_w, sm_h, anchor_num, 1, rpn_conf.lstm_t);
            bbox_inputs     = zeros(sm_w, sm_h, anchor_num*4, 1, rpn_conf.lstm_t);
            bbox_w_inputs   = zeros(sm_w, sm_h, anchor_num*4, 1, rpn_conf.lstm_t);

            if rpn_conf.has_weak
                weak_seg_inputs     = zeros(sm_w, sm_h, 1, 1, rpn_conf.lstm_t);
                weak_weights_inputs = zeros(sm_w, sm_h, 1, 1, rpn_conf.lstm_t);
            end
            
            for timeind=1:rpn_conf.lstm_t
                    
                sampleind = lstm_inds(timeind);
                [net_inputs, ~, im_rgb, im_aug] = proposal_generate_minibatch(rpn_conf, image_roidb_train(sampleind));
                
                %subplot(1,1,1); imshow([im_rgb im_aug]); drawnow;
                
                % first seq should be 0 (reset), all others 1
                seq_inputs(1, timeind) = single(~(timeind==1));

                rpn_test_net = reshape_input_data(rpn_test_net, net_inputs(1));
                rpn_test_net.forward(net_inputs(1));
                layer_data = rpn_test_net.blobs(rpn_conf.lstm_layer).get_data();

                data_inputs(:,:,:, 1, timeind)     = layer_data;
                labels_inputs(:,:,:, 1, timeind)   = net_inputs{2};
                labels_w_inputs(:,:,:, 1, timeind) = net_inputs{3};
                bbox_inputs(:,:,:, 1, timeind)     = net_inputs{4};
                bbox_w_inputs(:,:,:, 1, timeind)   = net_inputs{5};

                if rpn_conf.has_weak
                    weak_seg_inputs(:,:,:, 1, timeind)       = net_inputs{6};
                    weak_weights_inputs(:,:,:, 1, timeind)   = net_inputs{7};
                end

            end
            
            net_inputs = { data_inputs, seq_inputs, labels_inputs, labels_w_inputs, bbox_inputs, bbox_w_inputs };
            
            if rpn_conf.has_weak
               net_inputs{end+1} = weak_seg_inputs;
               net_inputs{end+1} = weak_weights_inputs;
            end

            % set input & step
            caffe_solver = reshape_input_data(caffe_solver, net_inputs, ~safeDefault(rpn_conf,'has_lstm'));
            caffe_solver = set_input_data(caffe_solver, net_inputs, ~safeDefault(rpn_conf,'has_lstm'));
            caffe_solver.step(1);

            % check loss
            cur_results = check_loss_rpn(rpn_conf, caffe_solver, cur_results);
            rpn_conf.iter = caffe_solver.iter();
            
        % recursive
        elseif rpn_conf.use_recursive
            
            [net_inputs, ~, im_rgb] = proposal_generate_minibatch(rpn_conf, image_roidb_train(sampleinds), rpn_test_net);
            
            % random cropping
            if isfield(rpn_conf, 'crop_size') && rpn_conf.crop_size(1)>size(net_inputs{1},2)
                
                dataH = size(net_inputs{1},2);
                dataW = size(net_inputs{1},1);
                targH = rpn_conf.test_scale;
                targW = round(targH*dataW/dataH);
                
                % if less than crop size, then at least match AR
                if isfield(rpn_conf, 'crop_size') && rpn_conf.crop_size(1)> dataH
                    targH = dataH;
                    targW = round(targH*rpn_conf.crop_size(2)/rpn_conf.crop_size(1));
                elseif isfield(rpn_conf, 'crop_size')
                    targH = rpn_conf.crop_size(1);
                    targW = rpn_conf.crop_size(2);
                end
                
                rows = randi(dataH-targH+1)+(0:targH-1);
                cols = randi(dataW-targW+1)+(0:targW-1);
                
                r1 = min(rows); r2 = max(rows);
                c1 = min(cols); c2 = max(cols);
                
                net_inputs{1} = net_inputs{1}(cols,rows,:);
                
                for netind=2:length(net_inputs)
                   
                    stride = dataH/size(net_inputs{netind},2);
                    
                    r1_tmp = ceil(r1/stride); r2_tmp = floor(r2/stride);
                    c1_tmp = ceil(c1/stride); c2_tmp = floor(c2/stride);
                    
                    net_inputs{netind} = net_inputs{netind}(c1_tmp:c2_tmp,r1_tmp:r2_tmp,:);
                    
                    assert(ceil(targH/stride) == size(net_inputs{netind},2));
                    assert(ceil(targW/stride) == size(net_inputs{netind},1));
                end
                
            end
            
            caffe_solver = set_input_data(caffe_solver, net_inputs);
            caffe_solver.step(1);
            
            if debug_im
                anchors = visualizeAnchors(net_inputs{2});
                anchors = imresize(anchors, [size(net_inputs{1},2), size(net_inputs{1},1)]);
                anchors_ign = visualizeAnchors(net_inputs{2}==255);
                anchors_ign = imresize(anchors_ign, [size(net_inputs{1},2), size(net_inputs{1},1)]);
                det = caffe_solver.net.blobs('proposal_cls_score_reshape').get_data(); 
                det = reshape(det(:,:,end), size(net_inputs{2},1), size(net_inputs{2},2), []);
                det = imresize(visualizeAnchors(det), [size(net_inputs{1},2), size(net_inputs{1},1)]);
                imshow([imresize(im_rgb,[size(net_inputs{1},2), size(net_inputs{1},1)]); anchors; anchors_ign; det]);
                drawnow;
            end
            
            % check loss
            cur_results = check_loss_rpn(rpn_conf, caffe_solver, cur_results);
            rpn_conf.iter = caffe_solver.iter();

        % Non-temporal
        else
            
            [net_inputs, ~, im_rgb] = proposal_generate_minibatch(rpn_conf, image_roidb_train(sampleinds));
            
            % dynamic IoU on?
            if safeDefault(rpn_conf, 'dynamic_iou') && rpn_conf.iter>rpn_conf.dynamic_start
                caffe_solver = set_input_data(caffe_solver, net_inputs);
                caffe_solver.net.forward(net_inputs);
                [net_inputs, ~, im_rgb] = proposal_generate_minibatch(rpn_conf, image_roidb_train(sampleinds), caffe_solver.net);
            end
            
            if rpn_conf.has_nms_lstm
               net_inputs = {net_inputs{:}, nms_lstm_inputs{:}};
            end
            
            
            % set input & step
            if ~rpn_conf.has_nms_lstm
                caffe_solver = reshape_input_data(caffe_solver, net_inputs, ~(safeDefault(rpn_conf,'has_lstm') || rpn_conf.has_nms_lstm));
            end
            
            
            caffe_solver = set_input_data(caffe_solver, net_inputs, ~(safeDefault(rpn_conf,'has_lstm') || rpn_conf.has_nms_lstm));
            caffe_solver.step(1);
            
            % check loss
            cur_results = check_loss_rpn(rpn_conf, caffe_solver, cur_results);
            rpn_conf.iter = caffe_solver.iter();
            
        end
        
        % -- print stats --
        if mod(rpn_conf.iter, rpn_conf.display_iter)==0
            
            loss_str = '';
            
            for lossind=1:length(rpn_conf.loss_layers)
        
                loss_name = rpn_conf.loss_layers{lossind};
                loss_val = mean(cur_results.(loss_name));
                
                loss_str = [loss_str sprintf('%s %.3g', strrep(loss_name, 'loss_',''), loss_val)];
                if lossind ~= length(rpn_conf.loss_layers), loss_str = [loss_str ', ']; end
                
                if ~isfield(all_results, loss_name), all_results.(loss_name) = []; end
                all_results.(loss_name) = [all_results.(loss_name); loss_val];
                cur_results.(loss_name) = [];
            end
            
            if ~isfield(all_results, 'acc'),    all_results.acc    = []; end
            if ~isfield(all_results, 'fg_acc'), all_results.fg_acc = []; end
            if ~isfield(all_results, 'bg_acc'), all_results.bg_acc = []; end
            
            all_results.acc    = [all_results.acc    mean(cur_results.acc)];
            all_results.fg_acc = [all_results.fg_acc mean(cur_results.fg_acc)];
            all_results.bg_acc = [all_results.bg_acc mean(cur_results.bg_acc)];
            
            cur_results.acc    = [];
            cur_results.fg_acc = [];
            cur_results.bg_acc = [];
            
            dt = toc/(rpn_conf.display_iter); tic;
            timeleft = max(dt*(rpn_conf.max_iter - rpn_conf.iter),0);
            if timeleft > 3600, timeleft = sprintf('%.1fh', timeleft/3600);
            elseif timeleft > 60, timeleft = sprintf('%.1fm', timeleft/60);
            else timeleft = sprintf('%.1fs', timeleft); end
            
            fprintf('Iter %d, acc %.2f, fg_acc %.2f, bg_acc %.2f, loss (%s), dt %.2f, eta %s\n', ...
                rpn_conf.iter, all_results.acc(end), all_results.fg_acc(end), ...
                all_results.bg_acc(end), loss_str, dt, timeleft);
            
            update_diary();
            
        end
        
        % -- test net --
        if mod(rpn_conf.iter, rpn_conf.snapshot_iter)==0 && rpn_conf.doVal
            
            rpn_conf.scales = rpn_conf.test_scale;
            
            % do not embedd bbox means/std?
            if rpn_conf.manual_bbox_means
                snapped_file = [rpn_conf.weights_dir '/' sprintf('snap_iter_%d.caffemodel', rpn_conf.iter)];
            % embed bbox means/std
            else
                snapped_file = write_snapshot(rpn_conf, caffe_solver, sprintf('snap_iter_%d.caffemodel', rpn_conf.iter));
            end
            
            results_dir = [rpn_conf.output_dir '/results/test_iter_' num2str(round(rpn_conf.iter/1000)) 'k'];
            solverstate_path = [rpn_conf.output_dir '/weights/snap_iter_' num2str(rpn_conf.iter)];
            
            reset_caffe(rpn_conf);
            
            % restore solver
            caffe_solver = caffe.get_solver(rpn_conf.solver_path);
            caffe_solver.restore([solverstate_path '.solverstate']);
            
        end
        
        % -- plot graphs --
        if rpn_conf.show_plot && mod(rpn_conf.iter, rpn_conf.display_iter)==0

            x = (rpn_conf.display_iter+start_iter):rpn_conf.display_iter:rpn_conf.iter;

            % loss plot
            subplot(1,2,1);

            plot(x,all_results.acc);
            hold on;
            plot(x,all_results.fg_acc);
            plot(x,all_results.bg_acc);
            legend('acc', 'fg-acc', 'bg-acc');
            hold off;

            % loss plot
            subplot(1,2,2);

            loss_legend = cell(length(rpn_conf.loss_layers),1);
            for lossind=1:length(rpn_conf.loss_layers)

                loss_name = rpn_conf.loss_layers{lossind};
                loss_legend{lossind} = strrep(loss_name, '_', '-');
                plot(x, all_results.(loss_name));
                hold on;
            end
            legend(loss_legend);
            hold off;

            drawnow;

        end
        
    end

    reset_caffe(rpn_conf);

    results_dir = [rpn_conf.output_dir '/results/test_iter_' num2str(round(rpn_conf.max_iter/1000)) 'k'];
    
    if ~exist(results_dir, 'file')
    
        % test net
        net = caffe.Net([rpn_conf.model_dir '/test.prototxt'], 'test');
        net.copy_from(snapped_file);

        fprintf('Processing final test for iter %d..', rpn_conf.max_iter);
        [mr, recall] = evaluate_results_rpn(rpn_conf, net, results_dir, rpn_conf.test_dir, rpn_conf.test_db);
        fprintf('mr %.4f, recall %.4f\n', mr, recall);

        clear net;
        clear caffe_solver;
    else
        [mr, ~, recall] = evaluate_result_dir({results_dir}, rpn_conf.test_db, rpn_conf.test_min_h);
        fprintf('mr %.4f, recall %.4f\n', mr, recall);
    end
    
    if ~exist(out_jargin.final_model_path, 'file')
        copyfile([rpn_conf.output_dir sprintf('/weights/snap_iter_%d.caffemodel', rpn_conf.max_iter)], out_jargin.final_model_path);
    end
    
    fprintf('Finished training rpn for %s.\n', config_name);
    
    update_diary();
    
end
