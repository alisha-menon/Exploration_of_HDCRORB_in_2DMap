function [out] = damv(data)
    data = detrend(data);
    out = sum(abs(diff(data)))/(length(data)-1);
end