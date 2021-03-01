function [prog_HVlist, distance_list, out, out_c, out_r, testAM_r, testAM_c, program_vec,condition_distance, result_distance, result_corrdist] = test_cross_validate_binary_RORB(model_r, model_c, features_r, features_c, N)
    numGests = size(features_c,1);
    numTrials = size(features_c,2);
    out = struct([]);
    out_r = struct([]);
    out_c = struct([]);
    
    configs = nchoosek(1:numTrials,N);
    numConfig = size(configs,1);
    testTrials = zeros(numConfig,numTrials-N);
    for i = 1:numConfig
        testTrials(i,:) = setdiff(1:numTrials,configs(i,:));
    end

    testAM_r = cell(numConfig,1);
    testAM_c = cell(numConfig,1);
    %testAM_c = cell(numConfig,1);
    result_AM = cell(numConfig,1);
    program_vec = cell(numConfig,1);
    distance_list = cell(numConfig,1);
    prog_HVlist = cell(numConfig,1);
    
    for i = 1:numConfig
        testAM_r{i} = containers.Map('KeyType','int32','ValueType','any');
        testAM_c{i} = containers.Map('KeyType','int32','ValueType','any');
        gestures = model_r.AM{1}.keys;
        for g = 1:length(gestures)
            testAM_r{i}(gestures{g}) = zeros(1,model_r.D);
            testAM_c{i}(gestures{g}) = zeros(1,model_c.D);
            for tr = 1:N
                testAM_r{i}(gestures{g}) = testAM_r{i}(gestures{g}) + model_r.AM{configs(i,tr)}(gestures{g});
                testAM_c{i}(gestures{g}) = testAM_c{i}(gestures{g}) + model_c.AM{configs(i,tr)}(gestures{g});
            end
        end
        %bipolarize_AM(testAM{i});
        binarize_AM(testAM_r{i});
        binarize_AM(testAM_c{i});
        %result_AM{i} = testAM_r{i};
        [program_vec{i}, distance_list{i}, prog_HVlist{i}] = program_memory(model_r, model_c, testAM_r{i}, testAM_c{i});
    end    
%     
%      trial = 1;
%      [testIdx,~] = find(testTrials == trial);
%      numTests = 1;
%      accuracy = zeros(1,g);
%      for g = 1:numGests
%          testData = features_c(g,trial).values;
%          testLabel = features_c(g,trial).label';
%  
%          testLength = size(testData,1);
%          sims = zeros(numTests, testLength, length(gestures));
%          outLabel = zeros(numTests, testLength);
%          
%          for i = 1:testLength-model_c.N+1
%                 segment_c = testData(i:i+model_c.N-1, :);
%                 ngram_c = ngram_iscas(segment_c, model_c);
%                 ngram_c = binarize_vec(ngram_c);
%                 protected_condition = circshift(ngram_c, [1,1]); 
%                 c = 1;
%                 noisy_resultHV = xor(protected_condition,program_vec{testIdx(c)});
%                 [sims(c,i,:), outLabel(c,i)] = classify_iscas_fast_hamming(noisy_resultHV, testAM_r{testIdx(c)});
%          end
%          out(c,g,trial).test = testLabel(1:end-model_c.N+1);
%          actual = testLabel(1:end-model_c.N+1);
%          predicted = outLabel(c,1:end-model_c.N+1);
%          out(c,g,trial).out = outLabel(c,1:end-model_c.N+1);
%          out(c,g,trial).sims = squeeze(sims(c,1:end-model_c.N+1,:));
%          out(c,g,trial).matches = sum(out(c,g,trial).test == out(c,g,trial).out);
%          out(c,g,trial).len = testLength-model_c.N+1;
%          out(c,g,trial).accuracy = out(c,g,trial).matches/out(c,g,trial).len;
%          accuracy(g) = out(c,g,trial).matches/out(c,g,trial).len;
%      end
    condition_distance = zeros(numGests,1);
    result_distance = zeros(numGests,1);
    result_corrdist = zeros(numGests,1);
    count_cond = zeros(numGests,1);    
    for trial = 1:numTrials
        [testIdx,~] = find(configs == trial);
        numTests = length(unique(testIdx));
        for g = 1:numGests
            testData = features_c(g,trial).values;
            testLabel = features_c(g,trial).label';
            testLength = size(testData,1);
            sims = zeros(numTests, testLength, length(gestures));
            outLabel = zeros(numTests, testLength);
            for i = 1:testLength-model_c.N+1
                segment_c = testData(i:i+model_c.N-1, :);
                ngram_c = ngram_iscas(segment_c, model_c);
                ngram_c = binarize_vec(ngram_c);
                protected_condition = circshift(ngram_c, [1,1]); 
                for c = 1:numTests
                    noisy_resultHV = xor(protected_condition,program_vec{testIdx(c)});
                    [sims(c,i,:), outLabel(c,i),min_resdist] = classify_iscas_fast_hamming(noisy_resultHV, testAM_r{testIdx(c)});
                    condition_distance(g) = condition_distance(g) + hamming_distance(ngram_c,testAM_c{testIdx(c)}(testLabel(1)));
                    count_cond(g) = count_cond(g) + 1;
                    result_distance(g) = result_distance(g) + min_resdist;
                    result_corrdist(g) = result_corrdist(g) + hamming_distance(noisy_resultHV, testAM_r{testIdx(c)}(testLabel(1)));
                end
            end
            
            for c = 1:numTests
                out(c,g,trial).test = testLabel(1:end-model_c.N+1);
                actual = testLabel(1:end-model_c.N+1);
                predicted = outLabel(c,1:end-model_c.N+1);
                out(c,g,trial).out = outLabel(c,1:end-model_c.N+1);
                out(c,g,trial).sims = squeeze(sims(c,1:end-model_c.N+1,:));
                out(c,g,trial).matches = sum(out(c,g,trial).test == out(c,g,trial).out);
                out(c,g,trial).len = testLength-model_c.N+1;
                out(c,g,trial).accuracy = out(c,g,trial).matches/out(c,g,trial).len;
                accuracy = out(c,g,trial).matches/out(c,g,trial).len;
            end
        end
    end
    for g = 1:numGests
        condition_distance(g) = int32(condition_distance(g)./count_cond(g));
        result_distance(g) = int32(result_distance(g)./count_cond(g));
        result_corrdist(g) = int32(result_corrdist(g)./count_cond(g));
    end
    for trial = 1:numTrials
        [testIdx,c] = find(configs == trial);
        numTests = length(unique(testIdx));
        for g = 1:numGests
            testData = features_c(g,trial).values;
            testLabel = features_c(g,trial).label';

            testLength = size(testData,1);
            sims = zeros(numTests, testLength, length(gestures));
            outLabel = zeros(numTests, testLength);

            for i = 1:testLength-model_c.N+1
                segment = testData(i:i+model_c.N-1, :);
                ngram = ngram_iscas(segment, model_c);
                ngram = binarize_vec(ngram);
                for c = 1:numTests
                    [sims(c,i,:), outLabel(c,i)] = classify_iscas_fast_hamming(ngram, testAM_c{testIdx(c)});
                end
            end
            
            for c = 1:numTests
                out_c(c,g,trial).test = testLabel(1:end-model_c.N+1);
                out_c(c,g,trial).out = outLabel(c,1:end-model_c.N+1);
                out_c(c,g,trial).sims = squeeze(sims(c,1:end-model_c.N+1,:));
                out_c(c,g,trial).matches = sum(out_c(c,g,trial).test == out_c(c,g,trial).out);
                out_c(c,g,trial).len = testLength-model_c.N+1;
                out_c(c,g,trial).accuracy = out_c(c,g,trial).matches/out_c(c,g,trial).len;
            end
        end
    end
    for trial = 1:numTrials
        [testIdx,c] = find(configs == trial);
        numTests = length(unique(testIdx));
        for g = 1:numGests
            testData = features_r(g,trial).values;
            testLabel = features_r(g,trial).label';

            testLength = size(testData,1);
            sims = zeros(numTests, testLength, length(gestures));
            outLabel = zeros(numTests, testLength);

            for i = 1:testLength-model_r.N+1
                segment = testData(i:i+model_r.N-1, :);
                ngram = ngram_iscas(segment, model_r);
                ngram = binarize_vec(ngram);
                for c = 1:numTests
                    [sims(c,i,:), outLabel(c,i)] = classify_iscas_fast_hamming(ngram, testAM_r{testIdx(c)});
                end
            end
            
            for c = 1:numTests
                out_r(c,g,trial).test = testLabel(1:end-model_r.N+1);
                out_r(c,g,trial).out = outLabel(c,1:end-model_r.N+1);
                out_r(c,g,trial).sims = squeeze(sims(c,1:end-model_r.N+1,:));
                out_r(c,g,trial).matches = sum(out_r(c,g,trial).test == out_r(c,g,trial).out);
                out_r(c,g,trial).len = testLength-model_r.N+1;
                out_r(c,g,trial).accuracy = out_r(c,g,trial).matches/out_r(c,g,trial).len;
            end
        end
    end
end