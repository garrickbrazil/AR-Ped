function imdata = visualizeBestLayer(layer)

    bestind = 1;
    bestval = mean(abs(reshape(layer(:,:,1), [], 1)));
    
    for layerind=1:size(layer,3)
    
        val = mean(abs(reshape(layer(:,:,layerind), [], 1)));
        if val>bestval
            bestind = layerind;
            bestval = val;
        end
    end

    imdata = layer(:,:,bestind);
    imdata = imdata - min(imdata(:));
    imdata = imdata / max(imdata(:));
    
end
