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
featureFunc =  @std;
model = struct;
model.D = 10000;
model.N = 5;
model.period = 500;
model.noCh = 64;

% save model parameters in log file
savelog(logfile,'Model parameters: \n');
savelog(logfile,['Feature function: ' func2str(featureFunc) '\n']);
savelog(logfile,['Model dimension: ' num2str(model.D) '\n']);
savelog(logfile,['Model N-length: ' num2str(model.N) '\n']);
savelog(logfile,['Model feature period: ' num2str(model.period) '\n']);
savelog(logfile,'\n');

%% Run for each subject
for subject = 2
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
    savelog(logfile,'Getting individual session accuracies: \n');
    for session = indivSessions
        gestures = gestList{session}; % use gestures associated with that particular session
        % gather data and extract features
        allData = exp{subject}{session};
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': gathering features\n']);
        features = get_features(allData, model.period, featureFunc);

        % reset model AM
        model = reset_AM(model,numTrials,gestures);

        % train the model, with a separate AM for each trial
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': training overall model\n']);
        model = train_model(model,features);

        % test the model with one shot
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, one-shot\n']);
        out = test_cross_validate(model,features,1);
        [actualGest, predictedGest, similarities, accTot] = get_stats(out);
        % save results
        savelog(logfile,'\n');
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
        savelog(logfile,'\n');
        save([outputDir 'Session' num2str(session) '_OneShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
        
%         % test the model with two shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, two-shot\n']);
%         out = test_cross_validate(model,features,2);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_TwoShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
        
        % test the model with three shot
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, three-shot\n']);
        out = test_cross_validate(model,features,3);
        [actualGest, predictedGest, similarities, accTot] = get_stats(out);
        % save results
        savelog(logfile,'\n');
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
        savelog(logfile,'\n');
        save([outputDir 'Session' num2str(session) '_ThreeShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
        
%         % test the model with four shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': testing model, four-shot\n']);
%         out = test_cross_validate(model,features,4);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Session ' num2str(session) ' accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_FourShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    end

    %% Baseline accuracy - all 21 gestures cross validated (experiment 1)
    gestures = allGest;
    % gather data and extract features
    allData = [exp{subject}{1}; exp{subject}{2}];
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': gathering features\n']);
    features = get_features(allData, model.period, featureFunc);

    % reset model AM
    model = reset_AM(model,numTrials,gestures);

    % train the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': training overall model\n']);
    model = train_model(model, features);
    
    % test the model one-shot
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, one-shot\n']);
    out = test_cross_validate(model,features,1);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    % save results
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'Baseline_OneShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     % test the model two-shot
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, two-shot\n']);
%     out = test_cross_validate(model,features,2);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     % save results
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Baseline_TwoShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    % test the model three-shot
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, three-shot\n']);
    out = test_cross_validate(model,features,3);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    % save results
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'Baseline_ThreeShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     % test the model four-shot
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': testing model, four-shot\n']);
%     out = test_cross_validate(model,features,4);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     % save results
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Baseline accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Baseline_FourShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')

    %% Arm position accuracy - single DOF gestures tested (experiments 1 and 2)
    gestures = singleDOF;
    % gather data and extract features
    trainData = exp{subject}{1};
    testData = exp{subject}{3};

    % gather features
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': gathering training features\n']);
    trainFeatures = get_features(trainData, model.period, featureFunc);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': gathering testing features\n']);
    testFeatures = get_features(testData, model.period, featureFunc);

    % reset model AM
    model = reset_AM(model,numTrials,gestures);

    % train the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': training overall old model\n']);
    model = train_model(model, trainFeatures);
    
    % testing the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, one-shot\n']);
    out = test_new_context(model,testFeatures,1);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'ArmPosition_OneShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, two-shot\n']);
%     out = test_new_context(model,testFeatures,2);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ArmPosition_TwoShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, three-shot\n']);
    out = test_new_context(model,testFeatures,3);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'ArmPosition_ThreeShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, four-shot\n']);
%     out = test_new_context(model,testFeatures,4);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ArmPosition_FourShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing old model, five-shot\n']);
%     out = test_new_context(model,testFeatures,5);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** five-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'ArmPosition_FiveShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    %% Arm position accuracy - incremental learning
    % train new model on new data
    model2 = struct;
    model2.D = model.D;
    model2.N = model.N;
    model2.period = model.period;
    model2.noCh = model.noCh;
    model2.eM = containers.Map ('KeyType','int32','ValueType','any');
    for e = 1:1:model2.noCh
        model2.eM(e) = model.eM(e);
    end
    
    model2 = reset_AM(model2,numTrials,gestures);

    % train the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': training model for new context\n']);
    model2 = train_model(model2, testFeatures);
    
    
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': testing combined models, 1 old, 1 new\n']);
    [outNew, outOld] = test_update_context(trainFeatures,testFeatures,model,model2,1,1);
    [actualGest, predictedGest, similarities, accTot] = get_stats(outNew);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** new context combined accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'ArmPositionCombined11New_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    [actualGest, predictedGest, similarities, accTot] = get_stats(outOld);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Arm position accuracy, subject ' num2str(subject) ': *** old context combined accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'ArmPositionCombined11Old_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    %% Different day accuracy - single DOF gestures tested (experiments 1 and 6)
    gestures = singleDOF;
    % gather data and extract features
    trainData = exp{subject}{1};
    testData = exp{subject}{7};

    % gather features
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': gathering training features\n']);
    trainFeatures = get_features(trainData, model.period, featureFunc);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': gathering testing features\n']);
    testFeatures = get_features(testData, model.period, featureFunc);

    % reset model AM
    model = reset_AM(model,numTrials,gestures);

    % train the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': training overall old model\n']);
    model = train_model(model, trainFeatures);
    
    % testing the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, one-shot\n']);
    out = test_new_context(model,testFeatures,1);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'DifferentDay_OneShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, two-shot\n']);
%     out = test_new_context(model,testFeatures,2);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'DifferentDay_TwoShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, three-shot\n']);
    out = test_new_context(model,testFeatures,3);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'DifferentDay_ThreeShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, four-shot\n']);
%     out = test_new_context(model,testFeatures,4);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'DifferentDay_FourShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing old model, five-shot\n']);
%     out = test_new_context(model,testFeatures,5);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** five-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'DifferentDay_FiveShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    %% Different day accuracy - incremental learning
    % train new model on new data
    model2 = struct;
    model2.D = model.D;
    model2.N = model.N;
    model2.period = model.period;
    model2.noCh = model.noCh;
    model2.eM = containers.Map ('KeyType','int32','ValueType','any');
    for e = 1:1:model2.noCh
        model2.eM(e) = model.eM(e);
    end
    
    model2 = reset_AM(model2,numTrials,gestures);

    % train the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': training model for new context\n']);
    model2 = train_model(model2, testFeatures);
    
    
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': testing combined models, 1 old, 1 new\n']);
    [outNew, outOld] = test_update_context(trainFeatures,testFeatures,model,model2,1,1);
    [actualGest, predictedGest, similarities, accTot] = get_stats(outNew);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** new context combined accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'DifferentDayCombined11New_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    [actualGest, predictedGest, similarities, accTot] = get_stats(outOld);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Different day accuracy, subject ' num2str(subject) ': *** old context combined accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'DifferentDayCombined11Old_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    %% Prolong accuracy - single DOF gestures tested (experiments 1 and 2)
    gestures = singleDOF;
    % gather data and extract features
    trainData = exp{subject}{7};
    testData = exp{subject}{8};

    % gather features
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': gathering training features\n']);
    trainFeatures = get_features(trainData, model.period, featureFunc);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': gathering testing features\n']);
    testFeatures = get_features(testData, model.period, featureFunc);

    % reset model AM
    model = reset_AM(model,numTrials,gestures);

    % train the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': training overall old model\n']);
    model = train_model(model, trainFeatures);
    
    % testing the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, one-shot\n']);
    out = test_new_context(model,testFeatures,1);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'Prolong_OneShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, two-shot\n']);
%     out = test_new_context(model,testFeatures,2);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Prolong_TwoShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, three-shot\n']);
    out = test_new_context(model,testFeatures,3);
    [actualGest, predictedGest, similarities, accTot] = get_stats(out);
    savelog(logfile,'\n');
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'Prolong_ThreeShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, four-shot\n']);
%     out = test_new_context(model,testFeatures,4);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Prolong_FourShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing old model, five-shot\n']);
%     out = test_new_context(model,testFeatures,5);
%     [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%     savelog(logfile,'\n');
%     savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** five-shot accuracy = ' num2str(accTot*100) '%%\n']);
%     savelog(logfile,'\n');
%     save([outputDir 'Prolong_FiveShot_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    %% Prolong accuracy - incremental learning
    % train new model on new data
    model2 = struct;
    model2.D = model.D;
    model2.N = model.N;
    model2.period = model.period;
    model2.noCh = model.noCh;
    model2.eM = containers.Map ('KeyType','int32','ValueType','any');
    for e = 1:1:model2.noCh
        model2.eM(e) = model.eM(e);
    end
    
    model2 = reset_AM(model2,numTrials,gestures);

    % train the model
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': training model for new context\n']);
    model2 = train_model(model2, testFeatures);
    
    
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': testing combined models, 1 old, 1 new\n']);
    [outNew, outOld] = test_update_context(trainFeatures,testFeatures,model,model2,1,1);
    [actualGest, predictedGest, similarities, accTot] = get_stats(outNew);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** new context combined accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'ProlongCombined11New_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    [actualGest, predictedGest, similarities, accTot] = get_stats(outOld);
    savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- ' 'Prolong accuracy, subject ' num2str(subject) ': *** old context combined accuracy = ' num2str(accTot*100) '%%\n']);
    savelog(logfile,'\n');
    save([outputDir 'ProlongCombined11Old_' num2str(subject)],'gestures','trainFeatures','testFeatures','model','model2','accTot','actualGest','predictedGest','similarities','-v7.3')
    
    %% Effort levels - effort gestures tested with experiments 3, 4, and 5
    effortLow = exp{subject}{6};
    effortMed = exp{subject}{4};
    effortHigh = exp{subject}{5};
    
    testData = struct([]);
    testData(1).data = [effortLow; effortMed];
    testData(1).gestures = effortGest;
    testData(2).data = [effortLow; effortHigh];
    testData(2).gestures = effortGest;
    testData(3).data = [effortMed; effortHigh];
    testData(3).gestures = effortGest;
    testData(4).data = [effortLow; effortMed; effortHigh];
    testData(4).gestures = effortGest;

    effortMedSep = effortMed;
    for i = 1:size(effortMedSep,1)
        for j = 1:size(effortMedSep,2)
            oldLabel = effortMedSep(i,j).label;
            oldLabel(oldLabel ~= 0) = oldLabel(oldLabel ~= 0) + 200;
            effortMedSep(i,j).label = oldLabel;
        end
    end

    effortHighSep = effortHigh;
    for i = 1:size(effortHighSep,1)
        for j = 1:size(effortHighSep,2)
            oldLabel = effortHighSep(i,j).label;
            oldLabel(oldLabel ~= 0) = oldLabel(oldLabel ~= 0) + 400;
            effortHighSep(i,j).label = oldLabel;
        end
    end

    testData(5).data = [effortLow; effortMedSep];
    testData(5).gestures = [effortGest effortGest+200];

    testData(6).data = [effortLow; effortHighSep];
    testData(6).gestures = [effortGest effortGest+400];

    testData(7).data = [effortMedSep; effortHighSep];
    testData(7).gestures = [effortGest+200 effortGest+400];

    testData(8).data = [effortLow; effortMedSep; effortHighSep];
    testData(8).gestures = [effortGest effortGest+200 effortGest+400];
    
    testName = {'low + med (same)';
        'low + high (same)';
        'med + high (same)';
        'low + med + high (same)';
        'low + med (separate)';
        'low + high (separate)';
        'med + high (separate)';
        'low + med + high (separate)'};

    for test = 1:length(testData)
        % get features for data
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': gathering features\n']);
        features = get_features(testData(test).data, model.period, featureFunc);

        % reset model AM
        model = reset_AM(model,numTrials,testData(test).gestures);

        % train the model
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': training overall model\n']);
        model = train_model(model, features);
        
        % test the model with one shot
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, one-shot\n']);
        out = test_cross_validate(model,features,1);
        [actualGest, predictedGest, similarities, accTot] = get_stats(out);
        % save results
        savelog(logfile,'\n');
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** one-shot accuracy = ' num2str(accTot*100) '%%\n']);
        savelog(logfile,'\n');
        save([outputDir 'Session' num2str(session) '_OneShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
        
%         % test the model with two shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, two-shot\n']);
%         out = test_cross_validate(model,features,2);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** two-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_TwoShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
        
        % test the model with three shot
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, three-shot\n']);
        out = test_cross_validate(model,features,3);
        [actualGest, predictedGest, similarities, accTot] = get_stats(out);
        % save results
        savelog(logfile,'\n');
        savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** three-shot accuracy = ' num2str(accTot*100) '%%\n']);
        savelog(logfile,'\n');
        save([outputDir 'Session' num2str(session) '_ThreeShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
        
%         % test the model with four shot
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': testing model, four-shot\n']);
%         out = test_cross_validate(model,features,4);
%         [actualGest, predictedGest, similarities, accTot] = get_stats(out);
%         % save results
%         savelog(logfile,'\n');
%         savelog(logfile,[datestr(now, 'HH:MM:SS') ' -- Accuracy for ' testName{test} ', subject ' num2str(subject) ': *** four-shot accuracy = ' num2str(accTot*100) '%%\n']);
%         savelog(logfile,'\n');
%         save([outputDir 'Session' num2str(session) '_FourShot_' num2str(subject)],'gestures','features','model','accTot','actualGest','predictedGest','similarities','-v7.3')
    end
end

%% Clean up
fclose(logfile);