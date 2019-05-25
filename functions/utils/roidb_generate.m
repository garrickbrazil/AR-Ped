function roidb = roidb_generate(imdb, flip, cache_dir, dataset, min_gt_height, conf)
% roidb = roidb_generate(imdb, flip)
%   Package the roi annotations into the imdb. 
%
%   Inspired by Ross Girshick's imdb and roidb code.

% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

mkdir_if_missing([cache_dir]);

roidb.name = imdb.name;

anno_path = ['./datasets/' dataset '/' roidb.name '/annotations'];

addpath(genpath('./external/code3.2.1'));
if conf.squarify
    pLoad={'lbls',conf.lbls,'ilbls',conf.ilbls,'squarify',{3,.41}};
else
    pLoad={'lbls',conf.lbls,'ilbls',conf.ilbls};
end

if isfield(conf, 'min_ol')
    pLoad = [pLoad 'hRng',[min_gt_height inf], 'vRng',[conf.min_ol 1] ];
else
    pLoad = [pLoad 'hRng',[min_gt_height inf], 'vRng',[1 1] ];
end
if flip
    cache_file = [cache_dir '/roidb_' dataset '_' imdb.name '_flip'];
else
    cache_file = [cache_dir '/roidb_' dataset '_' imdb.name];
end

cache_file = [cache_file, '.mat'];

try
  load(cache_file);
  fprintf('Preloaded roidb %s.. ', roidb.name);
catch

  fprintf('Computing roidb %s.. ', roidb.name);
  roidb.name = imdb.name;

  regions = [];

  if isempty(regions)
      regions.boxes = cell(length(imdb.image_ids), 1);
  end
  
  height = imdb.sizes(1,1);
  width = imdb.sizes(1,2);
  files=bbGt('getFiles',{anno_path});
  num_gts = 0;
  num_gt_no_ignores = 0;
  for i = 1:length(files)
      
      [objs,gts]=bbGt('bbLoad',files{i},pLoad);
      ignores = gts(:,end);
      num_gts  = num_gts + length(ignores);
      num_gt_no_ignores  = num_gt_no_ignores + (length(ignores)-sum(ignores));
      
      if flip
          % for ori
          x1 = gts(:,1);
          y1 = gts(:,2);
          x2 = gts(:,1) + gts(:,3);
          y2 = gts(:,2) + gts(:,4);
          gt_boxes = [x1 y1 x2 y2];
          roidb.rois(i*2-1) = attach_proposals(regions.boxes{i}, gt_boxes, ignores, objs, conf);
          
          % for flip
          x1_flip = width - gts(:,1) - gts(:,3);
          y1_flip = y1;
          x2_flip = width - gts(:,1);
          y2_flip = y2;
          gt_boxes_flip = [x1_flip y1_flip x2_flip y2_flip];
          roidb.rois(i*2) = attach_proposals(regions.boxes{i}, gt_boxes_flip, ignores, objs, conf);
          
          if 0
              % debugging visualizations
              im = imread(imdb.image_at(i*2-1));
              t_boxes = roidb.rois(i*2-1).boxes;
              for k = 1:size(t_boxes, 1)
                  showboxes2(im, t_boxes(k,1:4));
                  title(sprintf('%s, ignore: %d\n', imdb.image_ids{i*2-1}, roidb.rois(i*2-1).ignores(k)));
                  pause;
              end
              im = imread(imdb.image_at(i*2));
              t_boxes = roidb.rois(i*2).boxes;
              for k = 1:size(t_boxes, 1)
                  showboxes2(im, t_boxes(k,1:4));
                  title(sprintf('%s, ignore: %d\n', imdb.image_ids{i*2}, roidb.rois(i*2).ignores(k)));
                  pause;
              end
          end
      else
          % for ori
          x1 = gts(:,1);
          y1 = gts(:,2);
          x2 = gts(:,1) + gts(:,3);
          y2 = gts(:,2) + gts(:,4);
          gt_boxes = [x1 y1 x2 y2];
          
          roidb.rois(i) = attach_proposals(regions.boxes{i}, gt_boxes, ignores, objs, conf); 
      end
  end
  
  save(cache_file, 'roidb', '-v7.3');
  
  %fprintf('num_gt / num_ignore %d / %d \n', num_gt_no_ignores, num_gts);
end
fprintf('done\n');
end

% ------------------------------------------------------------------------
function rec = attach_proposals(boxes, gt_boxes, ignores, objs, conf)
% ------------------------------------------------------------------------

%           gt: [2108x1 double]
%      overlap: [2108x20 single]
%      dataset: 'voc_2007_trainval'
%        boxes: [2108x4 single]
%         feat: [2108x9216 single]
%        class: [2108x1 uint8]


all_boxes = cat(1, gt_boxes, boxes);
gt_classes = ones(size(gt_boxes, 1), 1); % set pedestrian label as 1
gt_igncls = ones(size(gt_boxes, 1), 1); % set pedestrian label as 1
num_gt_boxes = size(gt_boxes, 1);

num_boxes = size(boxes, 1);

for gtind=1:size(gt_classes, 1)
    clsmatch=1;
    clsnotfound = true;
    for clsind=1:length(conf.lbls)
       if strcmpi(objs(gtind).lbl, conf.lbls{clsind})
          clsmatch = clsind;
          clsnotfound = false;
          break;
       end
    end
    gt_classes(gtind) = clsmatch;
    gt_igncls(gtind)  = clsnotfound;
end

rec.gt = cat(1, true(num_gt_boxes, 1), false(num_boxes, 1));
rec.overlap = zeros(num_gt_boxes+num_boxes, length(conf.lbls), 'single');
for i = 1:num_gt_boxes
  rec.overlap(:, gt_classes(i)) = ...
      max(rec.overlap(:, gt_classes(i)), boxoverlap(all_boxes, gt_boxes(i, :)));
end
rec.boxes = single(all_boxes);
rec.feat = [];
rec.class = uint8(cat(1, gt_classes, zeros(num_boxes, 1)));
rec.igncls = gt_igncls;
rec.ignores = ignores;
end
