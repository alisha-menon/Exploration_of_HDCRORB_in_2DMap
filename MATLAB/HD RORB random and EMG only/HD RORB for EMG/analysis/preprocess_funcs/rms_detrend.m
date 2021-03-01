function [out] = rms_detrend(data)
    data = detrend(data);
    out = rms(data);
end