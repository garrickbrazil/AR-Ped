function conf = config_func()
    
    conf.model                  =  'VGG16_weak_seg';
    conf.dataset_train          =  'caltechx10';
    conf.dataset_test           =  'caltechx1';
    conf.dataset_val            =  'caltechval';
    conf.ext                    =  '.jpg';
    
    % solver
    conf.solver_type            =  'SGD';
    conf.lr                     =  0.001;
    conf.step_size              =  60000;
    conf.max_iter               =  20000;
    conf.snapshot_iter          =  5000;

    % general    
    conf.display_iter           =  1000;
    conf.rng_seed               =  3;
    conf.mat_rng_seed           =  3;
    conf.fg_thresh              =  0.7;
    conf.image_means            = [123.6800, 116.7790, 103.9390];

    % network settings
    conf.train_batch_size       =  20;        % number of proposals train
    conf.test_batch_size        =  20;        % number of proposals test
    conf.crop_size              =  [112 112]; % size of images
    conf.has_weak               =  true;      % has weak segmentation?
    conf.weak_seg_crop          =  [7 7];     % weak segmentation size
    conf.feat_stride            =  16;        % network stride
    conf.cost_sensitive         =  true;      % use cost sensitive
    conf.cost_mean_height       =  50;        % cost sensitive mean
    conf.fg_image_ratio         =  0.5;       % percent fg images
    conf.batch_size             =  120;       % number fg boxes
    conf.natural_fg_weight      =  true;      % ignore fg_fraction!
    conf.fg_fraction            =  1/5;       % percent fg boxes
    conf.feat_scores            =  true;      % fuse feature scores of rpn
    conf.padfactor              =  0.2;       % percent padding
    
    conf.use_best_only          = 0;
    conf.soft_suppress         = 1;
    
    conf.big_batch              = 4;
    conf.has_bbox_reg           = 0;
    conf.custom_feat_cls_layer  = 'proposal_cls_score_reshape3';
    conf.concat_radial_grad     = 0;
    conf.radial_layers          = {'proposal_cls_score3'};
    
    conf.hard_thres             = 0.005;
    
    %conf.pretrained             = 'vgg16.caffemodel';
    
    conf.lbls                   =  {'person'};
    conf.ilbls                  =  {'people', 'ignore'};
    conf.squarify               =  true;
    conf.use_best               = false;
    conf.attach_best            = false;
    conf.stage1_importance      = false;
    conf.contrastive            = false;
    
    conf.has_weak2              = 0;
    conf.has_weak3              = 0;    
    
    %% testing
    conf.test_db                = 'UsaTest';     % dataset to test with
    conf.val_db                 = 'UsaTrainVal'; % dataset to test with
    conf.min_gt_height          =  30;           % smallest gt to train on
    conf.test_min_h             =  50;           % database setting for min gt
    
    % temporal settings
    conf.use_temporal_in        =  false;     % disable/enable temporal in
    conf.temporal_layer         =  'im';      % conv layer to use (im = image)
    conf.temporal_im            =  true;      % is temporal input images?
    conf.temporal_num           =  3;         % number of timesteps
    
    conf.image_means = reshape(conf.image_means, [1 1 3]);

end
