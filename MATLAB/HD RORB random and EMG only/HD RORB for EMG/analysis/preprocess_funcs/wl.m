function [out] = wl(data)
    data = detrend(data);
    out = sum(abs(diff(data)));
end