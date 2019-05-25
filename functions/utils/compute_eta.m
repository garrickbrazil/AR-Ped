function [timeleft, dt] = compute_eta(timespent, ind, total)

    dt = timespent/ind;
    timeleft = max(dt*(total - ind),0);
    if timeleft > 3600, timeleft = sprintf('%.1fh', timeleft/3600);
    elseif timeleft > 60, timeleft = sprintf('%.1fm', timeleft/60);
    else timeleft = sprintf('%.1fs', timeleft); end

end
