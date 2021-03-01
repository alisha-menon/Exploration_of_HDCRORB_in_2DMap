%{
Experiments:
- baseline: all 21 gestures cross validated, experiment 1 (ex = 1,2)
- arm position: single DOF gestures trained on experiment 1 (ex = 1) and
tested on experiment 2 (ex = 3)
- effort level: effort gesture subset tested for three different levels,
with low (experiment 5, ex = 6), medium (experiment 3, ex = 4), and high
(experiment 4, ex = 5) efforts
- different day: single DOF gestures trained on experiment 1 (ex = 1) and
tested on experiment 6 (ex = 7)
- prolong: single DOF gestures trained on experiment 6 (ex = 7) and
tested on experiment 7 (ex = 8)
- rotate: single DOF gestures trained on experiment 6 (ex = 7) and
tested on experiment 8 (ex = 9)
%}

%% Reset workspace
close all
clear
clc

addpath(genpath('.'))

%% Load data
load('./info/info.mat')
exp = cell(1,1);
% exp{1} = load_subject_data(1);
exp{2} = load_subject_data(2);
% exp{3} = load_subject_data(3);
% exp{4} = load_subject_data(4);
clearvars i dataDir

%% Create output directory and log file
runtime = datestr(now,'yyyy-mm-dd_HH-MM-SS');
outputDir = ['./techcon/logs/' runtime '/'];
mkdir(outputDir)
logfile = fopen([outputDir 'log.txt'],'a');

%% Create model
% condition
featureFunc =  @std;
model_c = struct;
model_c.D = 10000;
model_c.N = 5;
model_c.period = 500;
model_c.noCh = 32;
% result
model_r = struct;
model_r.D = 10000;
model_r.N = 5;
model_r.period = 500;
model_r.noCh = 32;
%test
model = struct;
model.D = 10000;
model.N = 5;
model.period = 500;
model.noCh = 64;


%r_channels = [2,4,6,8,10,12,14,16,9:16,21:24,61:64];
%r_channels = [[9,32,27,8,20,7,14,10,19,29,6,13,23,24,15,26,30,16,31,11,2,12,22,17,28,25,3,1,5,21,18,4]];

% save model parameters in log file
savelog(logfile,'Model parameters: \n');
savelog(logfile,['Feature function: ' func2str(featureFunc) '\n']);
savelog(logfile,['Model dimension: ' num2str(model_c.D) '\n']);
savelog(logfile,['Model N-length: ' num2str(model_c.N) '\n']);
savelog(logfile,['Model feature period: ' num2str(model_c.period) '\n']);
savelog(logfile,'\n');

%% Run for each subject
for subject = 2
    load('channel_scores_alisha.mat')
    %for fisher score
    score_type = 1;
    %for tree-based feature importance
    %score_type = 2;
    channel_importance = allScores(:,subject,score_type);
    [~, ordered] = sort(channel_importance,'descend');    
    %c_channels = [ordered(1:2:64)];
    %r_channels = [ordered(2:2:64)];
    c_channels = [1:2:64];
    r_channels = [2:2:64];
    model_c.eM = containers.Map ('KeyType','int32','ValueType','any');
    % make item memory for condition and result channels
    for e = 1:1:model_c.noCh
        if ismember(e,subjectInfo(subject).exclude)
            model_c.eM(e) = zeros(1,model_c.D);
        else
            model_c.eM(e) = gen_random_HV(model_c.D);
        end
    end
    model_r.eM = containers.Map ('KeyType','int32','ValueType','any');
    for e = 1:1:model_r.noCh
        if ismember(e,subjectInfo(subject).exclude)
            model_r.eM(e) = zeros(1,model_r.D);
        else
            model_r.eM(e) = gen_random_HV(model_r.D);
        end
    end
    model.eM = containers.Map ('KeyType','int32','ValueType','any');
    for e = 1:1:model.noCh
        if ismember(e,subjectInfo(subject).exclude)
            model.eM(e) = zeros(1,model.D);
        else
            model.eM(e) = gen_random_HV(model.D);
        end
    end
    savelog(logfile,'\n');
    savelog(logfile,['Running subject ' num2str(subject)]);
    savelog(logfile,'\n');
    %% Get accuracies for individual sessions
%     indivSessions = [1 2 3 7 8];
    indivSessions = [1 2 3 4 5 6 7 8 9];
    accuracy = zeros(size(indivSessions,2), 3);
    prog_vector_dist = zeros(size(indivSessions,2), 1);
    prog_mean_distance = zeros(size(indivSessions,2),1);
%    savelog(logfile,'Getting individual session accuracies: \n');
     for session = indivSessions
        gestures = gestList{session}; % use gestures associated with that particular session
%        % gather data and extract features
        allData = exp{subject}{session};
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': gathering features\n']);
        features = get_features(allData, model.period, featureFunc);
        features_c = struct([]);
        features_r = struct([]);
        for h = 1:1:size(features,1)
            for w = 1:1:size(features,2)
                features_c(h,w).values = features(h,w).values(:,c_channels);
                features_c(h,w).label = features(h,w).label;
                features_r(h,w).values = features(h,w).values(:,r_channels);
                features_r(h,w).label = features(h,w).label;
            end
        end
    
        % reset model AM
        model_c = reset_AM(model_c,numTrials,gestures);
        model_r = reset_AM(model_r,numTrials,gestures);
        model = reset_AM(model,numTrials,gestures);

        % train the model
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': training overall model\n']);
        %% start here
        model_c = train_model(model_c, features_c);
        model_r = train_model(model_r, features_r);
        model = train_model(model, features);
    
        % test the model one-shot
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, one-shot\n']);
%         [out_3_bin,model,testAM_3_bin] = test_cross_validate_binary(model,features,3);
%         [out_1_bin,model,testAM_1_bin] = test_cross_validate_binary(model,features,1);
%        [out_c_bin,model_c,testAM_c_bin] = test_cross_validate_binary(model_c,features,3);
%        [out_r_bin,model_r,testAM_r_bin] = test_cross_validate_binary(model_r,features,3);
%         [actualGest, predictedGest, similarities, accTot_3_bin] = get_stats(out_3_bin);
%         [actualGest, predictedGest, similarities, accTot_1_bin] = get_stats(out_1_bin);
%        [actualGest, predictedGest, similarities, accTot_c_bin] = get_stats(out_c_bin);
%        [actualGest, predictedGest, similarities, accTot_r_bin] = get_stats(out_r_bin);
        
%        result_AM = testAM_r_bin{1};
%        program_vec = program_memory(model_r, model_c, testAM_r_bin{1}, testAM_c_bin{1});
        
        [prog_HVlist, distance_list, out_final, out_c, out_r, testAM_r, testAM_c, program_vec, condition_distance, result_distance, result_corrdist] = test_cross_validate_binary_RORB(model_r, model_c, features_r, features_c, 1);
        [actualGest, predictedGest, similarities, accTot] = get_stats(out_final);
        [actualGest_r, predictedGest_r, similarities_r, accTot_r] = get_stats(out_r);
        [actualGest_c, predictedGest_c, similarities_c, accTot_c] = get_stats(out_c);
        accuracy(session, :) = [accTot'; accTot_r'; accTot_c'];
%         C = confusionmat(actualGest,predictedGest);
%         confusionchart(C,gestures)
        % make confusion matrix with hamming distance between resultAM
        Hamming_conf1 = zeros(size(gestures,2));
        Hamming_conf2 = zeros(size(gestures,2));
        for w = 1:1:size(testAM_r,1)
            for i = 1:1:size(gestures,2)
                for h = 1:1:size(gestures,2)
                    Hamming_conf1(i,h) = (result_distance(i) > hamming_distance(testAM_r{w}(gestures(i)),testAM_r{w}(gestures(h))));
                    Hamming_conf2(i,h) = Hamming_conf2(i,h) + hamming_distance(testAM_r{w}(gestures(i)),testAM_r{w}(gestures(h)));
                end
            end
        end
        Hamming_conf2 = int32(Hamming_conf2./size(testAM_r,1));
        %figure
        %confusionchart(Hamming_conf2,gestures)
        %figure
        %confusionchart(Hamming_conf1,gestures)
        Total_squares = sum(Hamming_conf1, 'all');
        %distance between each vector and program memory
        distance_average = zeros(1,size(testAM_r,1));
        for i = 1:1:size(testAM_r,1)
            distance_average(i) = mean(distance_list{i});
        end
        prog_vector_dist(session) = mean(distance_average);
        prog_vector_sim(session) = 10000 - prog_vector_dist(session);
        
        %distance between each vector in program memory
        prog_conf = zeros(size(gestures,2));
        for w = 1:1:size(testAM_r,1)
            for i = 1:1:size(gestures,2)
                for h = 1:1:size(gestures,2)
                    prog_conf(i,h) = prog_conf(i,h) + hamming_distance(prog_HVlist{w}(i,:),prog_HVlist{w}(h,:));
                end
            end
        end
        prog_conf = int32(prog_conf./size(testAM_r,1));
        %figure 
        %confusionchart(prog_conf,gestures)
        prog_mean_distance(session) = mean2(prog_conf);
        prog_mean_similarity = 10000 - prog_mean_distance;
        %[out_3,model,testAM_3] = test_cross_validate(model,features,3);
        %[out_1,model,testAM_1] = test_cross_validate(model,features,1);
        %[out_c,model_c,testAM_c] = test_cross_validate(model_c,features,3);
        %[out_r,model_r,testAM_r] = test_cross_validate(model_r,features,3);
        %[actualGest, predictedGest, similarities, accTot_3] = get_stats(out_3);
        %[actualGest, predictedGest, similarities, accTot_1] = get_stats(out_1);
        %[actualGest, predictedGest, similarities, accTot_c] = get_stats(out_c);
        %[actualGest, predictedGest, similarities, accTot_r] = get_stats(out_r);
        %         % reset model AM
%         model = reset_AM(model,numTrials,gestures);
% 
%         % train the model, with a separate AM for each trial
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': training overall model\n']);
%         model = train_model(model,features);
% 
%         % test the model with one shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, one-shot\n']);
%         out = test_cross_validate(model,features,1);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_OneShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%         
% %         % test the model with two shot
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, two-shot\n']);
% %         out = test_cross_validate(model,features,2);
% %         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %         % save results
% %         savelog(logfile,'\n');
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %         savelog(logfile,'\n');
% %         save([outputDir 'Session' num2str(session) '_TwoShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%         
%         % test the model with three shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, three-shot\n']);
%         out = test_cross_validate(model,features,3);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_ThreeShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%         
% %         % test the model with four shot
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, four-shot\n']);
% %         out = test_cross_validate(model,features,4);
% %         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %         % save results
% %         savelog(logfile,'\n');
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %         savelog(logfile,'\n');
% %         save([outputDir 'Session' num2str(session) '_FourShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
     end

%     %% Baseline accuracy - all 21 gestures cross validated (experiment 1)
%     gestures = allGest;
%     % gather data and extract features
%     allData = [exp{subject}{1}; exp{subject}{2}];
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': gathering features\n']);
%     features = get_features(allData, model_c.period, featureFunc);
%     features_c = struct([]);
%     features_r = struct([]);
%     for h = 1:1:size(features,1)
%         for w = 1:1:size(features,2)
%             features_c(h,w).values = features(h,w).values(:,c_channels);
%             features_c(h,w).label = features(h,w).label;
%             features_r(h,w).values = features(h,w).values(:,r_channels);
%             features_r(h,w).label = features(h,w).label;
%         end
%     end
%     
%     % reset model AM
%     model_c = reset_AM(model_c,numTrials,gestures);
%     model_r = reset_AM(model_r,numTrials,gestures);
%     model = reset_AM(model,numTrials,gestures);
% 
%     % train the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': training overall model\n']);
%     %% start here
%     model_c = train_model(model_c, features_c);
%     model_r = train_model(model_r, features_r);
%     model = train_model(model, features);
%     
%     % test the model one-shot
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, one-shot\n']);
%     [out_3,model,testAM_3] = test_cross_validate(model,features,3);
%     [out_1,model,testAM_1] = test_cross_validate(model,features,1);
%     [out_c,model_c,testAM_c] = test_cross_validate(model_c,features,3);
%     [out_r,model_r,testAM_r] = test_cross_validate(model_r,features,3);
%     [actualGest, predictedGest, similarities, accTot_3] = get_stats(out_3);
%     [actualGest, predictedGest, similarities, accTot_1] = get_stats(out_1);
%     [actualGest, predictedGest, similarities, accTot_c] = get_stats(out_c);
%     [actualGest, predictedGest, similarities, accTot_r] = get_stats(out_r);
%     % save results
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Baseline_OneShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     % test the model two-shot
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, two-shot\n']);
%     out = test_cross_validate(model,features,2);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     % save results
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Baseline_TwoShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     % test the model three-shot
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, three-shot\n']);
%     out = test_cross_validate(model_c,features,3);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     % save results
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Baseline_ThreeShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     % test the model four-shot
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, four-shot\n']);
%     out = test_cross_validate(model,features,4);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     % save results
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Baseline_FourShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')

%     %% Arm position accuracy - single DOF gestures tested (experiments 1 and 2)
%     gestures = singleDOF;
%     % gather data and extract features
%     trainData = exp{subject}{1};
%     testData = exp{subject}{3};
% 
%     % gather features
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': gathering training features\n']);
%     trainFeatures = get_features(trainData, model.period, featureFunc);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': gathering testing features\n']);
%     testFeatures = get_features(testData, model.period, featureFunc);
% 
%     % reset model AM
%     model = reset_AM(model,numTrials,gestures);
% 
%     % train the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': training overall old model\n']);
%     model = train_model(model, trainFeatures);
%     
%     % testing the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, one-shot\n']);
%     out = test_new_context(model,testFeatures,1);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ArmPosition_OneShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, two-shot\n']);
% %     out = test_new_context(model,testFeatures,2);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'ArmPosition_TwoShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, three-shot\n']);
%     out = test_new_context(model,testFeatures,3);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ArmPosition_ThreeShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, four-shot\n']);
% %     out = test_new_context(model,testFeatures,4);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'ArmPosition_FourShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, five-shot\n']);
% %     out = test_new_context(model,testFeatures,5);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** five-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'ArmPosition_FiveShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     %% Arm position accuracy - incremental learning
%     % train new model on new data
%     model2 = struct;
%     model2.D = model.D;
%     model2.N = model.N;
%     model2.period = model.period;
%     model2.noCh = model.noCh;
%     model2.eM = containers.Map ('KeyType','int32','ValueType','any');
%     for e = 1:1:model2.noCh
%         model2.eM(e) = model.eM(e);
%     end
%     
%     model2 = reset_AM(model2,numTrials,gestures);
% 
%     % train the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': training model for new context\n']);
%     model2 = train_model(model2, testFeatures);
%     
%     
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing combined models, 1 old, 1 new\n']);
%     [outNew, outOld] = test_update_context(trainFeatures,testFeatures,model,model2,1,1);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(outNew);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** new context combined accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ArmPositionCombined11New_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     [actualGest, predictedGest, similarities, accTot] = get_stats(outOld);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** old context combined accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ArmPositionCombined11Old_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     %% Different day accuracy - single DOF gestures tested (experiments 1 and 6)
%     gestures = singleDOF;
%     % gather data and extract features
%     trainData = exp{subject}{1};
%     testData = exp{subject}{7};
% 
%     % gather features
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': gathering training features\n']);
%     trainFeatures = get_features(trainData, model.period, featureFunc);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': gathering testing features\n']);
%     testFeatures = get_features(testData, model.period, featureFunc);
% 
%     % reset model AM
%     model = reset_AM(model,numTrials,gestures);
% 
%     % train the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': training overall old model\n']);
%     model = train_model(model, trainFeatures);
%     
%     % testing the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, one-shot\n']);
%     out = test_new_context(model,testFeatures,1);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'DifferentDay_OneShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, two-shot\n']);
% %     out = test_new_context(model,testFeatures,2);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'DifferentDay_TwoShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, three-shot\n']);
%     out = test_new_context(model,testFeatures,3);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'DifferentDay_ThreeShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, four-shot\n']);
% %     out = test_new_context(model,testFeatures,4);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'DifferentDay_FourShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, five-shot\n']);
% %     out = test_new_context(model,testFeatures,5);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** five-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'DifferentDay_FiveShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     %% Different day accuracy - incremental learning
%     % train new model on new data
%     model2 = struct;
%     model2.D = model.D;
%     model2.N = model.N;
%     model2.period = model.period;
%     model2.noCh = model.noCh;
%     model2.eM = containers.Map ('KeyType','int32','ValueType','any');
%     for e = 1:1:model2.noCh
%         model2.eM(e) = model.eM(e);
%     end
%     
%     model2 = reset_AM(model2,numTrials,gestures);
% 
%     % train the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': training model for new context\n']);
%     model2 = train_model(model2, testFeatures);
%     
%     
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing combined models, 1 old, 1 new\n']);
%     [outNew, outOld] = test_update_context(trainFeatures,testFeatures,model,model2,1,1);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(outNew);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** new context combined accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'DifferentDayCombined11New_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     [actualGest, predictedGest, similarities, accTot] = get_stats(outOld);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** old context combined accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'DifferentDayCombined11Old_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     %% Prolong accuracy - single DOF gestures tested (experiments 1 and 2)
%     gestures = singleDOF;
%     % gather data and extract features
%     trainData = exp{subject}{7};
%     testData = exp{subject}{8};
% 
%     % gather features
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': gathering training features\n']);
%     trainFeatures = get_features(trainData, model.period, featureFunc);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': gathering testing features\n']);
%     testFeatures = get_features(testData, model.period, featureFunc);
% 
%     % reset model AM
%     model = reset_AM(model,numTrials,gestures);
% 
%     % train the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': training overall old model\n']);
%     model = train_model(model, trainFeatures);
%     
%     % testing the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, one-shot\n']);
%     out = test_new_context(model,testFeatures,1);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Prolong_OneShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, two-shot\n']);
% %     out = test_new_context(model,testFeatures,2);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'Prolong_TwoShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, three-shot\n']);
%     out = test_new_context(model,testFeatures,3);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Prolong_ThreeShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, four-shot\n']);
% %     out = test_new_context(model,testFeatures,4);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'Prolong_FourShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, five-shot\n']);
% %     out = test_new_context(model,testFeatures,5);
% %     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %     savelog(logfile,'\n');
% %     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** five-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %     savelog(logfile,'\n');
% %     save([outputDir 'Prolong_FiveShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     %% Prolong accuracy - incremental learning
%     % train new model on new data
%     model2 = struct;
%     model2.D = model.D;
%     model2.N = model.N;
%     model2.period = model.period;
%     model2.noCh = model.noCh;
%     model2.eM = containers.Map ('KeyType','int32','ValueType','any');
%     for e = 1:1:model2.noCh
%         model2.eM(e) = model.eM(e);
%     end
%     
%     model2 = reset_AM(model2,numTrials,gestures);
% 
%     % train the model
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': training model for new context\n']);
%     model2 = train_model(model2, testFeatures);
%     
%     
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing combined models, 1 old, 1 new\n']);
%     [outNew, outOld] = test_update_context(trainFeatures,testFeatures,model,model2,1,1);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(outNew);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** new context combined accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ProlongCombined11New_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     [actualGest, predictedGest, similarities, accTot] = get_stats(outOld);
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** old context combined accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ProlongCombined11Old_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
%     
%     %% Effort levels - effort gestures tested with experiments 3, 4, and 5
%     effortLow = exp{subject}{6};
%     effortMed = exp{subject}{4};
%     effortHigh = exp{subject}{5};
%     
%     testData = struct([]);
%     testData(1).data = [effortLow; effortMed];
%     testData(1).gestures = effortGest;
%     testData(2).data = [effortLow; effortHigh];
%     testData(2).gestures = effortGest;
%     testData(3).data = [effortMed; effortHigh];
%     testData(3).gestures = effortGest;
%     testData(4).data = [effortLow; effortMed; effortHigh];
%     testData(4).gestures = effortGest;
% 
%     effortMedSep = effortMed;
%     for i = 1:size(effortMedSep,1)
%         for j = 1:size(effortMedSep,2)
%             oldLabel = effortMedSep(i,j).label;
%             oldLabel(oldLabel ~= 0) = oldLabel(oldLabel ~= 0) + 200;
%             effortMedSep(i,j).label = oldLabel;
%         end
%     end
% 
%     effortHighSep = effortHigh;
%     for i = 1:size(effortHighSep,1)
%         for j = 1:size(effortHighSep,2)
%             oldLabel = effortHighSep(i,j).label;
%             oldLabel(oldLabel ~= 0) = oldLabel(oldLabel ~= 0) + 400;
%             effortHighSep(i,j).label = oldLabel;
%         end
%     end
% 
%     testData(5).data = [effortLow; effortMedSep];
%     testData(5).gestures = [effortGest effortGest+200];
% 
%     testData(6).data = [effortLow; effortHighSep];
%     testData(6).gestures = [effortGest effortGest+400];
% 
%     testData(7).data = [effortMedSep; effortHighSep];
%     testData(7).gestures = [effortGest+200 effortGest+400];
% 
%     testData(8).data = [effortLow; effortMedSep; effortHighSep];
%     testData(8).gestures = [effortGest effortGest+200 effortGest+400];
%     
%     testName = {'low + med (same)';
%         'low + high (same)';
%         'med + high (same)';
%         'low + med + high (same)';
%         'low + med (separate)';
%         'low + high (separate)';
%         'med + high (separate)';
%         'low + med + high (separate)'};
% 
%     for test = 1:length(testData)
%         % get features for data
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': gathering features\n']);
%         features = get_features(testData(test).data, model.period, featureFunc);
% 
%         % reset model AM
%         model = reset_AM(model,numTrials,testData(test).gestures);
% 
%         % train the model
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': training overall model\n']);
%         model = train_model(model, features);
%         
%         % test the model with one shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, one-shot\n']);
%         out = test_cross_validate(model,features,1);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_OneShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%         
% %         % test the model with two shot
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, two-shot\n']);
% %         out = test_cross_validate(model,features,2);
% %         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %         % save results
% %         savelog(logfile,'\n');
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %         savelog(logfile,'\n');
% %         save([outputDir 'Session' num2str(session) '_TwoShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%         
%         % test the model with three shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, three-shot\n']);
%         out = test_cross_validate(model,features,3);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_ThreeShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%         
% %         % test the model with four shot
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, four-shot\n']);
% %         out = test_cross_validate(model,features,4);
% %         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
% %         % save results
% %         savelog(logfile,'\n');
% %         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
% %         savelog(logfile,'\n');
% %         save([outputDir 'Session' num2str(session) '_FourShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
%     end
end

%% Clean up
fclose(logfile);