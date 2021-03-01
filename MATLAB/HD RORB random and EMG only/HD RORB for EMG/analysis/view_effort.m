close all
clear
clc

exp = load_subject_data(1);
lsbuV = (100000)./(2^15);
low = detrend(exp{6}(9,5).raw).*lsbuV;
med = detrend(exp{4}(9,5).raw).*lsbuV;
high = detrend(exp{5}(9,5).raw).*lsbuV;

% figure
% plot([low; med; high])

% rms4 = (std(detrend(exp{4}(9,2).raw(4000:4500,:))));
% rms5 = (std(detrend(exp{5}(9,2).raw(4000:4500,:))));
% rms6 = (std(detrend(exp{6}(9,2).raw(4000:4500,:))));

figure
plot([low(:,12); NaN(1000,1); med(:,12); NaN(1000,1); high(:,12)])

figure
plot(low(:,12) + 600*lsbuV)
hold on
plot(med(:,12) + 300*lsbuV)
plot(high(:,12))

xlim([0 11000])
ylim([-300 800].*lsbuV)
yticks(-500:100:2200);
xticks(0:500:11000)
 
% lowRaw = detrend(exp{6}(9,5).raw);
% medRaw = detrend(exp{4}(9,5).raw);
% highRaw = detrend(exp{5}(9,5).raw);
% 
% N = length(lowRaw)/50 - 3;
% lowRMS = zeros(N,64);
% medRMS = lowRMS;
% highRMS = lowRMS;
% for i = 1:N
%     idx = (1:200) + (i-1)*50;
%     lowRMS(i,:) = std(detrend(lowRaw(idx,:)));
%     medRMS(i,:) = std(detrend(medRaw(idx,:)));
%     highRMS(i,:) = std(detrend(highRaw(idx,:)));
% end
% 
% lowRMS = (lowRMS - 2)*30/10;
% medRMS = (medRMS - 2)*30/10;
% highRMS = (highRMS - 2)*30/10;
% 
% figure 
% plot(lowRMS(:,12))
% hold on
% plot(medRMS(:,12))
% plot(highRMS(:,12))