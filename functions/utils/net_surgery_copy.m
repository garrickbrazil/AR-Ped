function net_surgery_copy(net_src, net_dst)

    for layerind=1:length(net_src.layer_names)
        
        layername = net_src.layer_names{layerind};
        paramcount = length(net_src.layers(layername).params);
        
        for paramind=1:paramcount
            w = net_src.layers(layername).params(paramind).get_data();            
            try
                net_dst.layers(layername).params(paramind).set_data(w);
            catch
                % do nothing
            end
        end
    end
end

