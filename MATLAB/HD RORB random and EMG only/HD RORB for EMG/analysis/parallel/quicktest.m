%% Reset workspace
close all
clear
clc
addpath(genpath('.'))
p = gcp('nocreate');
if isempty(p)
    p = parpool(4);
end
% spmd, mpiprofile('on'); end

%% Load data
load('./info/info.mat')
exp = {};

% sub = [1 2 3 4 5]; % subjects to test
sub = 1;

for s = sub
    exp{s} = load_subject_data(s);
end

%% Create output directory
runtime = datestr(now,'yyyy-mm-dd_HH-MM-SS');
outputDir = ['./parallel/test/' runtime '/'];
mkdir(outputDir)

%% Loop through subjects and model parameters
feat = {@rms_detrend, @damv, @dasdv,@mav, @mfl, @samp_en_wrapper, @wamp, @wl, @zero_crossing}; %Add features, various combinations
win = [50];
dim = [10000];
nLen = [5];

Nshot = [3];

% set up mesh for looping
[subGrid,featGrid,winGrid,dimGrid,nLenGrid] = ndgrid(sub,feat,win,dim,nLen);

numRuns = numel(subGrid);

%% Overall loop
tic
for ii = 1:numRuns
    % get subject
    subject = subGrid(ii);
    
    % create model
    model = struct;
    model.D = dimGrid(ii);
    model.N = nLenGrid(ii);
    model.period = winGrid(ii);
    model.noCh = 64;
    model.eM = containers.Map ('KeyType','int32','ValueType','any');
    for e = 1:1:model.noCh
        if ismember(e,subjectInfo(subject).exclude)
            model.eM(e) = zeros(1,model.D);
        else
            model.eM(e) = gen_random_HV(model.D);
        end
    end
    
    % select feature function
    featureFunc = featGrid{ii};
    funcStr = strrep(func2str(featureFunc),'_','-');
    if length(funcStr)>10
        funcStr = funcStr(1:10);
    end
    funcStr = pad(funcStr,10,'left','-');
    
    % get output file description
    fileDes = [sprintf('%03d',subject) '_' funcStr '_' sprintf('%03d',model.period) '_' sprintf('%02d',model.N) '_' sprintf('%05d',model.D)];
    fprintf([fileDes '\n'])
    fileDes = [outputDir fileDes];
    
    % run individual sessions
    session = 1;
    gestures = gestList{session};
    allData = exp{subject}{session};
    features = get_features(allData, model.period, featureFunc);
    outfname = [fileDes '-Session_' num2str(session)];
    jobs = parfeval(p,@crossvalidate_func,0,model,gestures,features,Nshot,outfname);
    for session = 2:9
        gestures = gestList{session};
        allData = exp{subject}{session};
        features = get_features(allData, model.period, featureFunc);
        outfname = [fileDes '-Session_' num2str(session)];
        jobs(end+1) = parfeval(p,@crossvalidate_func,0,model,gestures,features,Nshot,outfname);
    end
    
    % run baseline experiment
%     gestures = allGest;
%     allData = [exp{subject}{1}; exp{subject}{2}];
%     features = get_features(allData, model.period, featureFunc);
%     outfname = [fileDes '-Baseline_'];
%     jobs(end+1) = parfeval(p,@crossvalidate_func,0,model,gestures,features,Nshot,outfname);
    
    % run arm position experiment
%     gestures = singleDOF;
%     trainData = exp{subject}{1};
%     testData = exp{subject}{3};
%     trainFeatures = get_features(trainData, model.period, featureFunc);
%     testFeatures = get_features(testData, model.period, featureFunc);
%     outfname = [fileDes '-ArmPos___'];
%     jobs(end+1) = parfeval(p,@newcontext_func,0,model,gestures,trainFeatures,testFeatures,Nshot,outfname);
%     outfname = [fileDes '-ArmPosUpd'];
%     jobs(end+1) = parfeval(p,@updatecontext_func,0,model,gestures,trainFeatures,testFeatures,1,1,outfname);
%     
    % run different day experiment
%     gestures = singleDOF;
%     trainData = exp{subject}{1};
%     testData = exp{subject}{7};
%     trainFeatures = get_features(trainData, model.period, featureFunc);
%     testFeatures = get_features(testData, model.period, featureFunc);
%     outfname = [fileDes '-DiffDay__'];
%     jobs(end+1) = parfeval(p,@newcontext_func,0,model,gestures,trainFeatures,testFeatures,Nshot,outfname);
%     outfname = [fileDes '-DiffDayUp'];
%     jobs(end+1) = parfeval(p,@updatecontext_func,0,model,gestures,trainFeatures,testFeatures,1,1,outfname);
%     
    % run prolong experiment
%     gestures = singleDOF;
%     trainData = exp{subject}{7};
%     testData = exp{subject}{8};
%     trainFeatures = get_features(trainData, model.period, featureFunc);
%     testFeatures = get_features(testData, model.period, featureFunc);
%     outfname = [fileDes '-Prolong__'];
%     jobs(end+1) = parfeval(p,@newcontext_func,0,model,gestures,trainFeatures,testFeatures,Nshot,outfname);
%     outfname = [fileDes '-ProlongUp'];
%     jobs(end+1) = parfeval(p,@updatecontext_func,0,model,gestures,trainFeatures,testFeatures,1,1,outfname);

    % run effort level
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
%         gestures = testData(test).gestures;
%         allData = testData(test).data;
%         features = get_features(allData, model.period, featureFunc);
%         outfname = [fileDes '-EffoType' num2str(test)];
%         jobs(end+1) = parfeval(p,@crossvalidate_func,0,model,gestures,features,Nshot,outfname);
%     end
%     
     if ~wait(jobs)
         break
     end
end
t = toc;
% spmd, mpiprofile('viewer'); end
f = fopen([outputDir 'elapsed_time.txt'],'w');
fprintf(f ,[num2str(t) ' seconds']);
fclose(f);
delete(p);
exit;
