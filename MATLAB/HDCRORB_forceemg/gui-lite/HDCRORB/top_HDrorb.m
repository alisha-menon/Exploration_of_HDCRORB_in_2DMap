close all
clear all
%HD recall of reactive behavior
%data format - columns for each feature/channel of data 
%====Features and Label===
load('first_try_5.mat')
features=raw; 

%no labels for recall
%f_label_a_binary=data_all(:,215);
%f_label_v_binary=data_all(:,216);

%for only EMG and force sensors, use force sensor as condition to predict
%EMG
%% select = 0 for regular, select = 1 for known condition tracking in training
select = 0;
% few_all = 0 for the extra bundling vector to be a permuted version of the
% xor of 2 features (marginally better for larger number of channels)
% few_all = 1 for the extra bundling vector to be a permuted version of the
% xor of all the features (marginally better for smaller number of channels)
few_all = 0;
rng('shuffle');
%% set these:
%maxL_condition_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]; % # of vectors in CiM
%maxL_result_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200, 250, 300, 350, 400, 450, 500]; % # of vectors in CiM
% number of force sensors CiM vectors
%maxL_condition_list = [4]; % # of vectors in CiM
% number of EMG channels CiM vectors
%maxL_result_list = [32]; % # of vectors in CiM
% valid EMG channels
%count_list = 32;
%maxL_condition_list = 20;
%maxL_result_list = 20;
%num_channels = 64;
%num_result_channels_list = [1, 2, 3, 4, 5, 10, 20, 40, 60, 80, 100];
valid_EMG = [0:8,10:31,56,59, 60:63]+1;
valid_force = [32, 33, 34, 35]+1;
num_result_channels_list = size(valid_EMG,2); % # of EMG channels
num_condition_channels_list = size(valid_force,2); % # of force sensor channels
maxL_condition = 40;
maxL_result = 40;
windowSize = 50;
D = 10000; %dimension of the hypervectors
learningrate=0.5; % percentage of the dataset used to train the algorithm (in this case 0.5 because first half is one trial,
%second half is another trial
trial1_train = 3;
trial2_recall = 5;
accuracy_check = 0.07;

%% need to change this for EMG and force sensor data, maybe mav?
valid_indexes1 = find(crc == 0);
partition = find(crc == 1);
total = [valid_indexes1 partition];
total_plot = [valid_indexes1];
valid_indexes = sort(total);
data_all = raw(valid_indexes,:);
data_all_plot = raw(sort(total_plot),:);
%data_all = raw;
result_data = data_all(:,valid_EMG);
result_data_plot = data_all_plot(:,valid_EMG);
condition_data = data_all(:,valid_force);
condition_data_plot = data_all_plot(:,valid_force);
in_trial = 0;
result_trials = [];
condition_trials = [];
start = [];
stop = [];
for i = 2:size(condition_data,1)-1
    if (data_all(i-1,1) == 0)
        in_trial = 1;
        start = [start, i];
    elseif (data_all(i+1,1) == -1)
        in_trial = 0;
        stop = [stop, i];
    end
    %if (in_trial == 1)
    %    result_trials = [result_trials; result_data(i,:)];
    %    condition_trials = [condition_trials; condition_data(i,:)];
    %end
end
scaled_condition_data_plot = double(condition_data_plot)/double(max(max(condition_data_plot)))*200;
x_count = (1:length(scaled_condition_data_plot))/1000;
plot(x_count, scaled_condition_data_plot)
xlabel('time (s)')
ylabel('Measured signal (mV)')
title('Gradual grasping of cylinder exercise: force sensor signals')
ylim([80, 220])
legend('Pointer', 'Middle', 'Ring','Thumb')
figure
scaled_result_data_plot = double(result_data_plot)/double(max(max(result_data_plot)))*200;
x_count_r = (1:length(scaled_result_data_plot))/1000;
plot(x_count_r, scaled_result_data_plot)
xlabel('time (s)')
ylabel('Measured signal (mV)')
title('Gradual grasping of cylinder exercise: EMG sensor signals')
ylim([150, 200])
legend('Pointer', 'Middle', 'Ring','Thumb')
%%
numWin1 = floor((stop(trial1_train) - start(trial1_train) + 1)/windowSize);
label1 = zeros(numWin1,1);
features_result_training = zeros(numWin1, num_result_channels_list);
features_condition_training = zeros(numWin1, num_condition_channels_list);
for i = 1:numWin1
    label1(i) = (100/numWin1)*i;
    for ch = 1:num_result_channels_list
        features_result_training(i,ch) = mav(result_data(start(trial1_train) + (1:windowSize)+(i-1)*windowSize,ch));
    end
    for ch = 1:num_condition_channels_list
        features_condition_training(i,ch) = mav(condition_data(start(trial1_train) + (1:windowSize)+(i-1)*windowSize,ch));
    end
end
numWin2 = floor((stop(trial2_recall) - start(trial2_recall) + 1)/windowSize);
label2 = zeros(numWin2,1);
features_result_recall = zeros(numWin2, num_result_channels_list);
features_condition_recall = zeros(numWin2, num_condition_channels_list);
for i = 1:numWin2
    label2(i) = (100/numWin2)*i;
    for ch = 1:num_result_channels_list
        features_result_recall(i,ch) = mav(result_data(start(trial2_recall) + (1:windowSize)+(i-1)*windowSize,ch));
    end
    for ch = 1:num_condition_channels_list
        features_condition_recall(i,ch) = mav(condition_data(start(trial2_recall) + (1:windowSize)+(i-1)*windowSize,ch));
    end
end
figure
scaled_condition_data_plot = double(features_condition_training)/double(max(max(features_condition_training)))*200;
x_count = (1:length(scaled_condition_data_plot))/20;
plot(x_count, scaled_condition_data_plot)
xlabel('time (s)')
ylabel('Measured signal (mV)')
title('Gradual grasping of cylinder exercise: force sensor signals')
%ylim([80, 220])
legend('Pointer', 'Middle', 'Ring','Thumb')
figure
scaled_result_data_plot = double(features_result_training)/double(max(max(features_result_training)))*200;
x_count_r = (1:length(scaled_result_data_plot))/20;
plot(x_count_r, scaled_result_data_plot)
xlabel('time (s)')
ylabel('Measured signal (mV)')
title('Gradual grasping of cylinder exercise: EMG sensor signals')
%ylim([160, 170])
close all
%%
%count_list = length(features_condition/2);
vis_features_results_recall = features_result_recall/1000;
vis_features_results_training = features_result_training/1000;
vis_features_condition_recall = features_condition_recall/1000;
vis_features_condition_training = features_condition_training/1000;
%accuracy_list = zeros(length(count_list), min(length(maxL_condition_list), length(maxL_result_list)));
%error_due_to_repeat = zeros(length(count_list), min(length(maxL_condition_list), length(maxL_result_list)));
range_result_training = (max(features_result_training/1000) - min(features_result_training/1000));
range_condition_training = (max(features_condition_training/1000) - min(features_condition_training/1000));
range_condition_training = max(range_condition_training);
range_result_training = max(range_result_training);
range_result_recall = (max(features_result_recall/1000) - min(features_result_recall/1000));
range_result_recall = max(range_result_recall);
range_condition_recall = (max(features_condition_recall/1000) - min(features_condition_recall/1000));
range_condition_recall = max(range_condition_recall);
%features_condition_training = features_condition_training/1000;
%features_result_training = features_result_training/1000;
%features_condition_training = (features_condition_training-min(features_condition_training));
%features_result_training = (features_result_training-min(features_result_training));
%features_condition_training = features_condition_training/range_condition;
%features_result_training = features_result_training/range_result;
%features_result_training = int64(features_result_training*maxL_result);
%features_condition_training = int64(features_condition_training*maxL_condition);
features_condition_training_og = features_condition_training;
features_condition_training = uint64((features_condition_training-min(features_condition_training))/1000*maxL_condition/range_condition_training);
features_condition_recall = uint64((features_condition_recall-min(features_condition_training_og))/1000*maxL_condition/range_condition_training);
sum_condition_training = sum(features_condition_training,2);
sum_condition_recall = sum(features_condition_recall,2);
features_result_training_og = features_result_training;
features_result_training = uint64((features_result_training-min(features_result_training))/1000*maxL_result/range_result_training);
features_result_recall = uint64((features_result_recall-min(features_result_training_og))/1000*maxL_result/range_result_training);
features_result_training (features_result_training <=0) = 0;
features_result_recall (features_result_recall <=0) = 0;
features_result_training (features_result_training >= maxL_result) = maxL_result;
features_result_recall (features_result_recall >=maxL_result) = maxL_result;

sum_result_training = sum(features_result_training,2);
sum_result_recall = sum(features_result_recall,2);
features_result_training = double(features_result_training);
features_result_recall = double(features_result_recall);
features_condition_training = double(features_condition_training);
features_condition_recall = double(features_condition_recall);


for g = 1:max(length(num_result_channels_list), length(num_condition_channels_list))
%for g = 1:length(count_list)
    num_result_channels = num_result_channels_list(1);
    num_condition_channels = num_condition_channels_list(1);
    %count = count_list;

    %% works
%     count = count/10;
%     features_result = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     features_condition = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     for i = 2:1:count
%         features_result = [features_result; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     end
%     features_result_1 = features_result;
%     for i = 2:1:num_result_channels
%         features_result = [features_result features_result_1];
%     end
%     for i = 2:1:count
%         features_condition = [features_condition; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
%     end
%     features_condition_1 = features_condition;
%     for i = 2:1:num_condition_channels
%         features_condition = [features_condition features_condition_1];
%     end
    %features_result = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
    %features_condition = [0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1; 0.1; 0.2; 0.3; 0.4; 0.5; 0.6; 0.7; 0.8; 0.9; 1];
    % for i = 1:1:count
    %     features_condition = [features_condition; 1; 0.9; 0.8; 0.7; 0.6; 0.5; 0.4; 0.3; 0.2; 0.1];
    % end

    %% =======HDC============
    if (select == 1)
            HD_functions_HDrorb_knowncondition;     % load HD functions with known condition clause
    else
        HD_functions_HDrorb;     % load HD functions
    end
    learningFrac = learningrate; 
    %classes = 2; % level of classes
    %precision = 20; % no use
    %ngram = 10; % for temporal encode
    %precision_condition = maxL_condition / max(features_condition(:));
    %precision_result = maxL_result / max(features_result(:));

    %% Encoding
    % Also generates continuous item memory for each channel that can be used
    % for encoding multiple values, quantized to nearest value of 1 until max
    % signal value set by maxL. Can check EMG
    % classification CiM method given mav pre-processing. Can use different
    % quantization for force sensors and EMG sensors
    [chAMcondition, iMchcondition, chAMresult, iMchresult, CiMC_rows, CiMR_rows, iMR_rows, iMC_rows] = initItemMemories (D, maxL_condition+1, num_condition_channels, maxL_result+1, num_result_channels);
    %[iMchcondition, iMchresult] = initbipolarItemMemories (D, num_condition_channels, num_result_channels);
    %% designate training vs. recall data
    %features_condition_training = features_condition(1:floor(count*learningrate),:); %for now..
    %features_result_training = features_result(1:floor(count*learningrate),:); %for now..
    %features_condition_recall = features_condition(floor(count*learningrate)+1:count,:); %for now..
    %features_result_recall = features_result(floor(count*learningrate)+1:count,:); %for now..
    %quantized_result_training = int64 (features_result_training * precision_result);
    %quantized_condition_training = int64 (features_condition_training * precision_condition);
    %quantized_result_recall = int64 (features_result_recall * precision_result);
    %quantized_condition_recall = int64 (features_condition_recall * precision_condition);
    size_training = size(features_condition_training);
    length_training = size_training(1);
    size_recall = size(features_condition_recall);
    length_recall = size_recall(1);

    %% training
    fprintf ('HDC for RORB\n');
    if (select == 1)
        [prog_HV, prog_HVlist, result_AM, known_condition_integers, result_integers] = hdcrorbtrain (few_all, length_training, features_condition_training, features_result_training, chAMcondition, chAMresult, iMchcondition, iMchresult, D, precision_condition, precision_result, num_condition_channels, num_result_channels); 
    else
        [prog_HV, result_AM, condition_AM,prog_HVlist, progHV_AM] = hdcrorbtrain (few_all, range_result_training, range_condition_training, length_training, features_condition_training, features_result_training, chAMcondition, chAMresult, iMchcondition, iMchresult, D, num_condition_channels, num_result_channels); 
    end
    close all
    condition_rows = zeros(size(condition_AM,1),D);
    for i = 1:1:size(condition_AM,1)
        condition_rows(i,:) = condition_AM(i);
    end
    result_rows = zeros(size(result_AM,1),D);
    for i = 1:1:size(result_AM,1)
        result_rows(i,:) = result_AM(i);
    end
    different_result = sum(xor(result_rows(76,:),result_rows(1,:)))
    different_condition = sum(xor(condition_rows(1,:),condition_rows(190,:)))
    different_progHV = sum(xor(prog_HVlist(1,:),prog_HVlist(190,:)))
    
    [actuator_values, actuator_noise, condition_values, condition_noise, result_values, result_noise, progHV_values, progHV_noise] = hdcrorbpredict (progHV_AM, few_all, length_recall, prog_HV, result_AM, condition_AM, features_condition_recall, features_result_recall, chAMcondition, chAMresult, iMchcondition, iMchresult, D, num_condition_channels, num_result_channels); 
    expected = (1:1:length_recall)';
    
    %% check the number of repeat condition samples
%     non_repeat_errors = [];
     row_wrong = [];
     for i = 1:1:length_recall
        if (sum(expected(i) == actuator_values(i)))
            row_wrong = [row_wrong i];
        end
     end
%      [A,u,c] = unique(quantized_condition_training,'rows');
%      [n,~,~] = histcounts(c,numel(u));
%     repeat_bin_address = find(n > 1);
%     repeat_address_condition = u(repeat_bin_address);
%     repeat_values_condition = quantized_condition_training(repeat_address_condition,:);
%     [length_repeat, ~] = size(repeat_values_condition);
%     wrong_recall_values_condition = quantized_condition_recall(row_wrong,:);
%     match_count_condition = 0;
%     [length_wrong, ~] = size(wrong_recall_values_condition);
%     non_repeat_errors = [];
%     for i = 1:1:length_wrong
%         x = 0;
%         for o = 1:1:length_repeat
%             if (repeat_values_condition(o,:) == wrong_recall_values_condition(i,:))
%                 x = 1;
%             end
%         end
%         if (x == 1)
%             match_count_condition = match_count_condition+1;
%         else
%             non_repeat_errors = [non_repeat_errors; wrong_recall_values_condition(i,:)]; %#ok<AGROW>
%         end
%     end
%     [~,u,c] = unique(quantized_result_training,'rows');
%     [n,~,bin] = histcounts(c,numel(u));
%     repeat_bin_address = find(n > 1);
%     repeat_address_result = u(repeat_bin_address);
%     repeat_values_result = quantized_result_training(repeat_address_result,:);
%     [length_repeat_result, ~] = size(repeat_values_result);
%     match_count_result = 0;
%     wrong_recall_values_result = quantized_result_recall(row_wrong,:);
%     [length_wrong_result, ~] = size(wrong_recall_values_result);
%     for i = 1:1:length_wrong_result
%         for o = 1:1:length_repeat_result
%             x = x+(repeat_values_result(o,:) == wrong_recall_values_result(i,:));
%         end
%         if (x > 0)
%             match_count_result = match_count_result+1;
%         end
%     end
% 
%     %% check whether the result samples were the same for all of the repeat condition samples
%     result_repeat_count = zeros(length_repeat,4);
%     location = [];
%     for i = 1:1:length_repeat
%         for o = 1:1:length_training
%             x = (repeat_values_condition(i,:) == quantized_condition_training(o,:));
%             if (x == 1)
%                 location = [location o];
%             end
%         end
%         %location = find(quantized_condition_training == repeat_values_condition(i));
%         unique_values = unique(quantized_result_training(location),'rows');
%         if (length(unique_values) == length(location))
%             result_repeat_count(i,1) = length(location);
%             result_repeat_count(i,2) = 0;
%         else
%             result_repeat_count(i,1) = length(location);
%             result_repeat_count(i,2) = length(location) - length(unique_values);
%         end
%         result_repeat_count(i,3) = result_repeat_count(i,1) - result_repeat_count(i,2);
%         if (result_repeat_count(i,3) > 0)
%             result_repeat_count(i,4) = 0;
%         else
% 
%             result_repeat_count(i,4) = 1;
%         end
%     end
%     number_unique_condition = sum(result_repeat_count(:,4));
%     training_sample_count = count*learningrate;
%     recall_sample_count = count*learningrate;
% 
%     training_sample_count
%     num_condition_channels
%     num_result_channels
%     recall_sample_count; 
%     error_due_to_repeat(g, j) = length(row_wrong) == match_count_condition;

%     maxL_condition
%     maxL_result;
    expected_length = size(expected,1);
%     accuracy_count = 0;
%     for h = 1:1:expected_length
%         channel_match_count = 0;
%         for u = 1:1:num_result_channels
%             match = (expected(h,u) == actuator_values(h,u));
%             channel_match_count = channel_match_count + match;
%         end
%         if channel_match_count == num_result_channels
%             accuracy_count = accuracy_count + 1;
%         end
%     end
    %%
    %accuracy_2 = accuracy_count/expected_length*100
    fp_actuator_accuracy = 0;
    fp_condition_accuracy = 0;
    fp_result_accuracy = 0;
    fp_progHV_accuracy = 0;
    absolute_accuracy = sum(actuator_values==expected)/length_recall*100
    for i=1:length(actuator_values)
        if (actuator_values(i) <= (expected(i)+expected_length*accuracy_check))
            if (actuator_values(i) >= (expected(i)-expected_length*accuracy_check))
                fp_actuator_accuracy = fp_actuator_accuracy+1;
            end
        end
        if (condition_values(i) <= (expected(i)+expected_length*accuracy_check))
            if (condition_values(i) >= (expected(i)-expected_length*accuracy_check))
                fp_condition_accuracy = fp_condition_accuracy + 1;
            end
        end
        if (result_values(i) <= (expected(i)+expected_length*accuracy_check))
            if (result_values(i) >= (expected(i)-expected_length*accuracy_check))
                fp_result_accuracy = fp_result_accuracy + 1;
            end
        end
        if (progHV_values(i) <= (expected(i)+expected_length*accuracy_check))
            if (progHV_values(i) >= (expected(i)-expected_length*accuracy_check))
                fp_progHV_accuracy = fp_progHV_accuracy + 1;
            end
        end
    end
%% 
    confusion_result = zeros(size(result_AM,1),size(result_AM,1));
    for i = 1:size(result_AM,1)
        for x = 1:size(result_AM,1)
            confusion_result(i,x) = sum(xor(result_AM(i),result_AM(x)));
        end
    end
    imagesc(int64(confusion_result))
    colorbar
    title('hamming distance between vectors in resultAM')
    xlabel('effort vectors')
    ylabel('effort vectors')
    confusion_progHV = zeros(size(progHV_AM,1),size(progHV_AM,1));
    for i = 1:size(progHV_AM,1)
        for x = 1:size(progHV_AM,1)
            confusion_progHV(i,x) = sum(xor(progHV_AM(i),progHV_AM(x)));
        end
    end
    figure
    imagesc(int64(confusion_progHV))
    colorbar
    title('hamming distance between vectors before bundling in program memory')
    xlabel('effort vectors')
    ylabel('effort vectors')
    confusion_condition = zeros(size(condition_AM,1),size(condition_AM,1));
    for i = 1:size(condition_AM,1)
        for x = 1:size(condition_AM,1)
            confusion_condition(i,x) = sum(xor(condition_AM(i),condition_AM(x)));
        end
    end
    figure
    imagesc(int64(confusion_condition))
    colorbar
    title('hamming distance between condition vectors')
    xlabel('effort vectors')
    ylabel('effort vectors')
%%
    
    fp_actuator_accuracy = fp_actuator_accuracy/length_recall*100
    fp_condition_accuracy = fp_condition_accuracy/length_recall*100
    fp_result_accuracy = fp_result_accuracy/length_recall*100
    fp_progHV_accuracy = fp_progHV_accuracy/length_recall*100
    %X = categorical({'Condition','ProgramHV','Actuator'});
    %X = reordercats(X,{'Condition','ProgramHV','Actuator'});
    accuracies = [fp_condition_accuracy; fp_progHV_accuracy; fp_actuator_accuracy];
    noise = [mean(condition_noise); mean(progHV_noise); mean(actuator_noise)];
    %bar(X,Y)
    %ylim([0,100])
    %accuracy_list(g, j) = accuracy;
    %accuracy_list_2(g, j) = accuracy_2;
    [length_row_wrong, ~] = size(row_wrong);
    %fprintf('%d of the %d recall samples were wrong, %d of these matched the repeats in the condition samples where %d of the repeated condition values also had the same result value for all the samples, and %d of these matched the repeats in the result samples\n',length(row_wrong), length(quantized_result_recall), match_count_condition, number_unique_condition, match_count_result);
    fprintf('%d of the %d recall samples were correct\n',length(row_wrong), length_recall);
end

    
