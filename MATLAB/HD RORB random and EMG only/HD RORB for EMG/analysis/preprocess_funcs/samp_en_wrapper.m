function [out] = samp_en_wrapper(data)
    data = detrend(data);
    out = samp_en(2, 0.2*std(data), data, 1);
end