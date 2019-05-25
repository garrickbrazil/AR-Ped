function conf = config_func()
    
    conf.model                  =  'VGG16_ar_rpn';
    conf.dataset_train          =  'caltechx10';
    conf.dataset_test           =  'caltechx1';
    conf.dataset_val            =  'caltechval';
    
    % solver
    conf.solver_type            =  'SGD';
    conf.lr                     =  0.001;
    conf.step_size              =  60000;
    conf.max_iter               =  110000;
    conf.snapshot_iter          =  10000;
    conf.doVal                  =  true;

    % general    
    conf.display_iter           =  1000;
    conf.rng_seed               =  3;
    conf.mat_rng_seed           =  3;
    conf.train_scales           =  [720];
    conf.scales_prob            =  [1];
    conf.test_scale             =  720
    conf.max_size               =  Inf;
    conf.bg_thresh_hi           =  0.6;
    conf.bg_thresh_lo           =  0;
    conf.fg_thresh              =  0.6;    
    conf.pretrained             =  'vgg16.caffemodel';
    conf.image_means            = [123.6800, 116.7790, 103.9390];

    % network settings    
    conf.has_weak               =  1;     % has weak segmentation?
    conf.feat_stride            =  16;    % network stride
    conf.cost_sensitive         =  false; % use cost sensitive
    conf.cost_mean_height       =  50;    % cost sensitive mean
    conf.fg_image_ratio         =  0.5;   % percent fg images
    conf.batch_size             =  Inf;   % number fg boxes
    conf.fg_fraction            =  1/5;   % percent fg boxes
    
    conf.weak_strides           =  2.^(4:-1:2);
    conf.attach_fg2             =  1;
    conf.fg_thresh2             =  0.4;
    conf.fg_thresh3             =  0.5;
    conf.easy_fg_thresh2        =  0;
    
    conf.attach_best            =  1;
    conf.best_gt_num            =  1;

    % anchors
    conf.anchor_scales          =  1.6*(1.385.^(0:8));
    conf.anchor_ratios          =  0.41;
    conf.base_anchor_size       =  16;
    
    conf.use_recursive          =  0;     % to use recurseive or not
    conf.recursive_no_holes     =  true;  % prevent holes
    conf.recursive_area         =  0.8;   % min area needed for segmentation
    conf.recursive_update       =  20000; % iteration update freq
    conf.recursive_layer        =  'seg_softmax';
    conf.recursive_boxes        =  false;
    conf.recursive_start        =  20000;
    conf.recursive_weights      =  true;

    conf.lbls                   =  {'person'};
    conf.ilbls                  =  {'people', 'ignore'};
    conf.squarify               =  true;
    
    conf.dynamic_iou            =  0
    
    %% testing
    conf.test_min_box_height    =  50;           % min box height to keep
    conf.test_min_box_size      =  16;           % min box size to keep (w || h)
    conf.nms_per_nms_topN       =  10000;        % boxes before nms
    conf.nms_overlap_thres      =  0.5;          % nms threshold IoU
    conf.nms_after_nms_topN     =  40;           % boxes after nms
    conf.test_db                = 'UsaTest';     % dataset to test with
    conf.val_db                 = 'UsaTrainVal'; % dataset to test with
    conf.min_gt_height          =  30;           % smallest gt to train on
    conf.test_min_h             =  50;           % database setting for min gt
    
    % data augmentation
    % note: we restrict blur and noise from happening to the same image
    conf.mirror_prob            =  0.0;          % mirror probability  [0=disabled]
    conf.blur_prob              =  0.0;          % bluring probability [0=disabled] 
    conf.noise_prob             =  0.0;          % noise probability   [0=disabled]
    
    conf.manual_bbox_means      =  true;         % do means and stds live (no embedding)
    
    conf.image_means = reshape(conf.image_means, [1 1 3]);

end
