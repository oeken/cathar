function r = has_converged(buffer, smooth, threshold)            
    val = tsmovavg(buffer,'s',smooth,1);  % moving average
    val = val(smooth:end); % remove NaNs
    change = abs(diff(val) ./ val(1:end-1));  % relative differences    
    good = change < threshold;  % if they are all less then threshold
    r = min(good);  % then output 1 else 0
end