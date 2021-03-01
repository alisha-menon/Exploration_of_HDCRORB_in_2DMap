function [out] = dasdv(data)
    data = detrend(data);
    out = sqrt(sum(diff(data).^2)/(length(data)-1));
end