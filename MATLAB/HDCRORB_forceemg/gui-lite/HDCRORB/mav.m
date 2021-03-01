function [out] = mav(data)
    %data = detrend(single(data));
    out = sum(abs(data))/length(data);
end