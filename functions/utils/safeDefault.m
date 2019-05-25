function val = safeDefault(conf, field, val)
    
    if ~exist('val', 'var')
        val = false;
    end
    
    if isfield(conf, field)
        val = conf.(field);
    end

end
