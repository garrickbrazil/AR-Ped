function o = boxoverlap(det, ign)
% Compute the symmetric intersection over union overlap between a set of
% bounding boxes in a and a single bounding box in b.
%
% a  a matrix where each row specifies a bounding box
% b  a matrix where each row specifies a bounding box

% AUTORIGHTS
% -------------------------------------------------------
% Copyright (C) 2011-2012 Ross Girshick
% Copyright (C) 2008, 2009, 2010 Pedro Felzenszwalb, Ross Girshick
% 
% This file is part of the voc-releaseX code
% (http://people.cs.uchicago.edu/~rbg/latent/)
% and is available under the terms of an MIT-like license
% provided in COPYING. Please retain this notice and
% COPYING if you use this file (or a portion of it) in
% your project.
% -------------------------------------------------------

o = cell(1, size(ign, 1));
for i = 1:size(ign, 1)
    x1 = max(det(:,1), ign(i,1));
    y1 = max(det(:,2), ign(i,2));
    x2 = min(det(:,3), ign(i,3));
    y2 = min(det(:,4), ign(i,4));

    w = x2-x1+1;
    h = y2-y1+1;
    inter = w.*h;
    detarea = (det(:,3)-det(:,1)+1) .* (det(:,4)-det(:,2)+1);
    ignarea = (ign(i,3)-ign(i,1)+1) * (ign(i,4)-ign(i,2)+1);
    % intersection over union overlap
    o{i} = inter ./ (detarea);
    % set invalid entries to 0 overlap
    o{i}(w <= 0) = 0;
    o{i}(h <= 0) = 0;
end

o = cell2mat(o);
