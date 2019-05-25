function [mr, recall] = evaluate_results_rpn(conf, net, output_dir, test_dir, test_db)

    debug = 0;
    
    if (exist(output_dir, 'dir')), rmdir(output_dir, 's'); end
    
    imlist = dir([test_dir '/images/*.jpg']);
    prevpath = '';
    
    if safeDefault(conf, 'has_lstm')

        lstm_extras.net = caffe.Net(conf.test_plain_rpn_path, 'test');
        lstm_extras.net.copy_from(conf.init_weights);
    end
    
    if safeDefault(conf, 'has_history')

        hist_inputs = cell(conf.hist_len,1);
            
        % init with zeros for history
        for timeind=1:length(hist_inputs)
            sm_w       = 60;
            sm_h       = 45;
            anchor_num = size(conf.anchors,1);
            hist_data  = zeros(sm_w, sm_h, anchor_num*2, 1, 'single');
            hist_inputs{timeind} = hist_data;
        end
    end
        
    tic;
    count = 0;
    
    if debug, fig = figure; end
    
    for imind=1:length(imlist)
        
        [~, id] = fileparts(imlist(imind).name);
        
        reg = regexp(id, '(set\d\d)_(V\d\d\d)_I(\d\d\d\d\d)', 'tokens');
        setname = reg{1}{1};
        vname = reg{1}{2};
        iname = reg{1}{3};
        inum = str2num(iname) + 1;
        
        %if safeDefault(conf, 'step') && safeDefault(conf, 'test_folder_skiprate') && ~(mod(inum, conf.step*conf.test_folder_skiprate)==0)
        %    continue;
        %end
        
        curpath = [output_dir '/' setname '/' vname '.txt'];
        
        %imobj = imlist{imind};
        im = imread([test_dir '/images/' imlist(imind).name]);
        
        if safeDefault(conf, 'has_history') && ~strcmp(curpath, prevpath)

            hist_inputs = cell(conf.hist_len,1);

            % init with zeros for history
            for timeind=1:length(hist_inputs)
                sm_w       = 60;
                sm_h       = 45;
                anchor_num = size(conf.anchors,1);
                hist_data  = zeros(sm_w, sm_h, anchor_num*2, 1, 'single');
                hist_inputs{timeind} = hist_data;
            end
        end
        
        if safeDefault(conf, 'has_lstm')
            
            if ~strcmp(curpath, prevpath)
                lstm_extras.h_cls  = zeros(60, 45, 18, 1, 1);
                lstm_extras.c_cls  = zeros(60, 45, 18, 1, 1);
                lstm_extras.h_bbox = zeros(60, 45, 36, 1, 1);
                lstm_extras.c_bbox = zeros(60, 45, 36, 1, 1);
            end
            
            % detect bboxes using lstm extras
            [pred_boxes, scores] = proposal_im_detect(conf, net, im, lstm_extras);
            
            % update hidden states
            lstm_extras.seq    = 1; 
            lstm_extras.h_cls  = net.blobs('proposal_hout_cls').get_data();
            lstm_extras.c_cls  = net.blobs('proposal_cout_cls').get_data();
            lstm_extras.h_bbox = net.blobs('proposal_hout_bbox').get_data();
            lstm_extras.c_bbox = net.blobs('proposal_cout_bbox').get_data();
            
        elseif safeDefault(conf, 'has_history')
            
            [pred_boxes, scores] = proposal_im_detect(conf, net, im, hist_inputs);
            
            det_out = net.blobs('proposal_cls_score').get_data();
            
            hist_inputs(1,:) = [];
            hist_inputs{end+1} = det_out; 
            
        else
            [pred_boxes, scores] = proposal_im_detect(conf, net, im);
        end
        
        aboxes = [pred_boxes scores];
        [aboxes, valid] = nms_filter(aboxes, conf.nms_per_nms_topN, conf.nms_overlap_thres, conf.nms_after_nms_topN , true);
        
        if ~strcmp(curpath, prevpath)
            
            if ~isempty(prevpath), fclose(fid); end
            
            mkdir_if_missing([output_dir '/' setname]);
            fid=fopen(curpath, 'a');
            prevpath = curpath;
            
        end
        
        for boxind=1:size(aboxes,1)
            
            x1 = (aboxes(boxind, 1));
            y1 = (aboxes(boxind, 2));
            x2 = (aboxes(boxind, 3));
            y2 = (aboxes(boxind, 4));
            score = aboxes(boxind, 5);
            w = x2 - x1 + 1;
            h = y2 - y1 + 1;
            if score >= 0.001 && (mod(inum,30)==0 || strcmpi(conf.test_db, 'kittitest'))
                fprintf(fid, '%d,%.3f,%.3f,%.3f,%.3f,%.3f\n', [inum x1 y1 w h score]);    
            end
            if debug && score > 0.90
                im = insertShape(im, 'rectangle', [x1 y1 w h], 'LineWidth', 3);
            end
        end
        
        if debug 
            
            anchorGrid = visualizeAnchors(net.blobs('proposal_cls_prob').get_data());
            im = imresize(im, [size(anchorGrid,1) size(anchorGrid,2)]);
            dispim = [im anchorGrid];
            
            if safeDefault(conf, 'has_lstm')
                hGrid = lstm_extras.h_cls - min(lstm_extras.h_cls(:));
                hGrid = hGrid/max(hGrid(:));
                hGrid = visualizeAnchors(hGrid);
                
                cGrid = lstm_extras.c_cls - min(lstm_extras.c_cls(:));
                cGrid = cGrid/max(cGrid(:));
                cGrid = visualizeAnchors(cGrid);
                
                dispim = [dispim hGrid cGrid];
            end
            
            imshow(dispim); 
            drawnow;  
        end
        
        count = count + 1;
        realdt = toc/count;
        
        dt = toc/(imind);
        timeleft = max(dt*(length(imlist) - imind),0);
        if timeleft > 3600, timeleft = sprintf('%.1fh', timeleft/3600);
        elseif timeleft > 60, timeleft = sprintf('%.1fm', timeleft/60);
        else timeleft = sprintf('%.1fs', timeleft); end
        if mod(count,100)==0,
            fprintf('%d/%d, dt=%.4f, eta=%s\n', imind, length(imlist), realdt, timeleft);
        end
    end
    
    if debug, close(fig); end
    
    fclose(fid);

    [mr, ~, recall] = evaluate_result_dir({output_dir}, test_db, conf.test_min_h);
    
end