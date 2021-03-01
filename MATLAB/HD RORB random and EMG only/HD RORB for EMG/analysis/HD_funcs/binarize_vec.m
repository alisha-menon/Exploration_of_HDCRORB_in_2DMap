function [out] = binarize_vec(vec)
    out = vec;
    numzeros = sum(out==0);
    filler = gen_HV_filler_binary(numzeros);
    out(out == 0) = filler;
    out(out > 0) = 1;
    out(out < 0) = 0;
end