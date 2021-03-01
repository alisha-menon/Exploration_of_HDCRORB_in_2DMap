function [out] = wamp(data)
    data = detrend(data);
    out = sum(abs(diff(data)) >= 8);
end