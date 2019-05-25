function im = augment_data(im, conf)
    
    % add blur?
    if conf.add_blur
        im = imgaussfilt(im, 0.2+rand()*0.3);
    end
    
    % add noise?
    if conf.add_noise
        im = imnoise(im, 'gaussian', 0, 0.0002+rand()*0.0003);
    end
   
end