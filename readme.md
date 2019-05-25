# Pedestrian Detection with Autoregressive Network Phases

Garrick Brazil, Xiaoming Liu

## Introduction

Pedestrian detection framework using autoregressive network phases as detailed in [arXiv report](https://arxiv.org/abs/1812.00440), accepted to CVPR 2019. 

Our AR-Ped framework is derivative work of [Faster R-CNN](https://github.com/ShaoqingRen/faster_rcnn), [RPN+BF](https://github.com/zhangliliang/RPN_BF), and [SDS-RCNN](https://github.com/garrickbrazil/SDS-RCNN). Tested with Ubuntu 16.04, CUDA 7.5, Matlab 2016a, Titan X GPU, and a modified version of Caffe v1.0 as provided. Unless otherwise stated the below scripts and instructions assume working directory in MATLAB is the project root. 


    @inproceedings{brazil2019pedestrian,
        title={Pedestrian Detection with Autoregressive Network Phases},
        author={Brazil, Garrick and Liu, Xiaoming},
        booktitle={Proceeding of IEEE Computer Vision and Pattern Recognition},
        address={Long Beach, CA},
        year={2019}
    }
    

## Setup

- **Build Caffe**

    Build caffe and matcaffe following the usual [instructions](http://caffe.berkeleyvision.org/installation.html). We provide an upgraded version of Caffe v1.0 which includes the required layers necessary to run Faster R-CNN in *external/caffe* (the same as in [SDS-RCNN](https://github.com/garrickbrazil/SDS-RCNN)).

- **Data**

    Download the full [Caltech](http://www.vision.caltech.edu/Image_Datasets/CaltechPedestrians/) dataset. In order to evaluate you must extract or soft-link a folder called *data-USA* into the directory *external/caltech_toolbox/* such that such that the annotation and video files can be accessed as: *data-USA/annotations/\*.vbb* and *data-USA/videos/\*.seq*.

    Then extract the datasets for train, val, test in Matlab as below (or setup softlinks as desire). 

    ```
    dbInfo('usatrain');    dbExtract('datasets/caltechx10/train', 1, 3);
    dbInfo('usatrainval'); dbExtract('datasets/caltechval/val', 1);
    dbInfo('usatest');     dbExtract('datasets/caltechx1/test', 1);
    ```
- **Misc**
    1. Download the pretrained [VGG16](https://www.cse.msu.edu/computervision/vgg16.zip) on ImageNet and place in *SDS-RCNN/pretrained/vgg16.caffemodel*.
    1. Run *build_nms* to compile nms mex files.
    1. Review the config files in *experiments/+Config/+[rcnn|rpn]* for additional information.

## Training

Training both stages takes about 18 hours on a single Titan X.

``` matlab
rpn_config  = 'caltech_VGG16_ar_rpn';
rcnn_config	= 'caltech_VGG16_weak_seg';
gpu_id = 1;

% train both stages
train_all(rpn_config, rcnn_config, gpu_id);
```

## Testing

We provide the the models for our AR-RPN (8.01% MR) and AR-Ped (6.45%). We further provide the necessary files of anchors, bbox_stds, bbox_means, and basic configurations which should be loaded into memory at test time as depicted below. All files are packed into [AR-Ped-Release.zip](https://www.cse.msu.edu/computervision/AR-Ped-Release.zip).

``` matlab
load('rpn_conf.mat');
load('rcnn_conf.mat');
load('anchors.mat');
load('bbox_means.mat');
load('bbox_stds.mat');
gpu_id = 1;

% test AR-RPN only
test_rpn(test_protsotxt_path, weights_path, rpn_conf, anchors, bbox_means, bbox_stds, gpu_id)

% test RPN and BCN (full AR-Ped)
test_rcnn(test_prototxt_path, weights_path, rpn_conf, anchors, bbox_means, bbox_stds, ...
    rcnn_prototxt, rcnn_weights, rcnn_conf, gpu_id)

```
