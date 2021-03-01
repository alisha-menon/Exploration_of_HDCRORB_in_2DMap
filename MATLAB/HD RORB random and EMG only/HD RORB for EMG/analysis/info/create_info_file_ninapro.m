close all
clear
clc

% singleDOF = 100:112;
singleDOF = 101:112;
multiDOF = 201:208;
allGest = [singleDOF multiDOF];
effortGest = [101 103 109 102 104 110 201 202 207];

numTrials = 5;
numTrain = 3;
numTest = 2;

trainTrials = nchoosek(1:numTrials,numTrain);
numIter = size(trainTrials,1);
testTrials = zeros(numIter,numTest);
for i = 1:size(trainTrials,1)
    testTrials(i,:) = setdiff(1:numTrials,trainTrials(i,:));
end

dataDir = '../gui/data/mat/';

subjectInfo = struct([]);

% ex = [1 1 2 3 4 5 6 7 8];
ex = [1];
gestList = {singleDOF}
%     multiDOF;
%     singleDOF;
%     effortGest;
%     effortGest;
%     effortGest;
%     singleDOF;
%     singleDOF;
%     singleDOF};

subjectInfo(1).dates = {'20180621';
    '20180621';
    '20180621';
    '20180621';
    '20180621';
    '20180621';
    '20180622';
    '20180622';
    '20180622'};
subjectInfo(1).extraEffort = true;
subjectInfo(1).exclude = [];

subjectInfo(2).dates = {'20180621';
    '20180621';
    '20180621';
    '20180515';
    '20180515';
    '20180515';
    '20180622';
    '20180622';
    '20180622'};
subjectInfo(2).extraEffort = false;
subjectInfo(2).exclude = [];

subjectInfo(3).dates = {'20180730';
    '20180730';
    '20180730';
    '20180730';
    '20180730';
    '20180730';
    '20180731';
    '20180731';
    '20180731'};
subjectInfo(3).extraEffort = true;
subjectInfo(3).exclude = 33:40;

subjectInfo(4).dates = {'20180801';
    '20180801';
    '20180801';
    '20180801';
    '20180801';
    '20180801';
    '20180802';
    '20180802';
    '20180802'};
subjectInfo(4).extraEffort = true;
subjectInfo(4).exclude = 33:40;

subjectInfo(5).dates = {'20180808';
    '20180808';
    '20180808';
    '20180808';
    '20180808';
    '20180808';
    '20180809';
    '20180809';
    '20180809'};
subjectInfo(5).extraEffort = true;
subjectInfo(5).exclude = 33:40;

subjectInfo(101).dates = {'20181212'};
subjectInfo(101).extraEffort = false;
subjectInfo(101).exclude = 11:64;

if exist('./info/info.mat', 'file')==2
  delete('./info/info.mat');
end
save('./info/info.mat')
    

