function [zc] = zero_crossing(data)
    data = detrend(data);
    a = data(1:end-1);
    b = data(2:end);
    zc = sum((a.*b < 0) & (abs(a - b) > 8));
end