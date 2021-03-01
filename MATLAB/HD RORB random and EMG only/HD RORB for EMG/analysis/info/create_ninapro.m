clear all
load('../../Ninapro/DB1/s1/S1_A1_E1.mat');

gestLen = 200;

toggles = (find(diff(restimulus)~=0));

for i = 1:length(toggles)/2
    centers(i) = toggles(2*i-1) + floor((toggles(2*i)-toggles(2*i-1))/2);
end

fnameBase = '../gui/data/mat/101/1/20181212-235959/101_1_';
keySet = 1:12;
valueSet = 100 + (1:12);
gestMap = containers.Map(keySet,valueSet);
gestMap(9) = 111;
gestMap(10) = 112;
gestMap(11) = 109;
gestMap(12) = 110;

for i=1:120
    data = emg(centers(i)-gestLen/2:centers(i)+gestLen/2-1,:);
    data = [data zeros(200,54)];
    fname = [fnameBase num2str(gestMap(restimulus(centers(i)))) '_20181212-235959_' num2str(rerepetition(centers(i))) '.mat'];
    gestLabel = ones(1,200)*gestMap(restimulus(centers(i)));
    streamInfo.subject = 101;
    streamInfo.rep = rerepetition(centers(i));
    streamInfo.timeGest = 200;
    save(fname,'data','gestLabel','streamInfo');
end
