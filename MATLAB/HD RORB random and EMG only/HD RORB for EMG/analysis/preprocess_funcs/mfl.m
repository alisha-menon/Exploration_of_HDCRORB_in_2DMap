function [out] = mfl(data)
    data = detrend(data);    
    out = log10(sqrt(sum(diff(data).^2)));
end