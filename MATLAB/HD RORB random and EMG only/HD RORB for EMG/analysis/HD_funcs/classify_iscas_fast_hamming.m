function [sims, label,minSim] = classify_iscas_fast_hamming(ngram, AM)
    classes = AM.keys;
    label = -1;
    minSim = 10000;
    sims = zeros(1,size(classes, 2));
    for i = 1:1:size(classes, 2)
        sims(i) = hamming_distance(AM(cell2mat(classes(i))), ngram);
        if sims(i) < minSim
            minSim = sims(i);
            label = cell2mat(classes(i));
        end
    end
end