close all
clear
clc

set(0,'DefaultAxesFontName','Helvetica')
set(0,'DefaultAxesFontSize',16)
set(0,'DefaultAxesTickDir','in')
set(0,'DefaultAxesLineWidth',1)
set(0,'DefaultAxesBox', 'off')

set(0,'DefaultLineLineWidth',2)
set(0,'DefaultLineLineStyle','-')
set(0,'DefaultLineMarkerSize',15)

set(0,'DefaultFigureColor',[1 1 1])

load('./parallel/outputs/aggregate_09_10_18');

colororder = [
	0.00  0.00  1.00
	0.00  0.50  0.00 
	1.00  0.00  0.00 
	0.00  0.75  0.75
	0.75  0.00  0.75
	0.75  0.75  0.00 
	0.25  0.25  0.25
	0.75  0.25  0.25
	0.95  0.95  0.00 
	0.25  0.25  0.75
	0.75  0.75  0.75
	0.00  1.00  0.00 
	0.76  0.57  0.17
	0.54  0.63  0.22
	0.34  0.57  0.92
	1.00  0.10  0.60
	0.88  0.75  0.73
	0.10  0.49  0.47
	0.66  0.34  0.65
	0.99  0.41  0.23
];

feat = {@zero_crossing, @wl, @wamp, @rms_detrend, @mfl, @mav, @dasdv, @damv};
win = [50 100 200];
nLen = [1 3 5];
types = {'Baseline';        %1
    'ArmPos___';            %2
    'ArmPosUpd-1-1-new';    %3
    'ArmPosUpd-1-1-old';    %4
    'DiffDay__';            %5
    'DiffDayUp-1-1-new';    %6
    'DiffDayUp-1-1-old';    %7
    'Prolong__';            %8
    'ProlongUp-1-1-new';    %9
    'ProlongUp-1-1-old';    %10
    'Session_1';            %11
    'Session_2';            %12
    'Session_3';            %13
    'Session_4';            %14
    'Session_5';            %15
    'Session_6';            %16
    'Session_7';            %17
    'Session_8';            %18
    'Session_9';            %19
    'EffoType1';            %20
    'EffoType2';            %21
    'EffoType3';            %22
    'EffoType4';            %23
    'EffoType5';            %24
    'EffoType6';            %25
    'EffoType7';            %26
    'EffoType8';            %27
    };

%% gather results
results = cell2struct(expnames,'names',2);
for i = 1:length(results)
    results(i).acc = meanAcc(i);
end
%% plot accuracies vs feature, window, nLen
baseIdx = find(contains(expnames,types{11}));
baseExp = expnames(baseIdx);
x = zeros(length(win),length(nLen));
y = zeros(length(win),length(nLen),length(feat));
for i = 1:length(win)
    x(i,:) = nLen.*win(i);
end

for f = 1:length(feat)
    funcStr = strrep(func2str(feat{f}),'_','-');
    if length(funcStr)>10
        funcStr = funcStr(1:10);
    end
    funcStr = pad(funcStr,10,'left','-');
    featIdx = baseIdx(find(contains(baseExp,funcStr)));
    
    exp = expnames(featIdx);
    acc = meanAcc(featIdx);
    
    for i = 1:length(win)
        for j = 1:length(nLen)
            expIdx = find(contains(exp,[sprintf('%03d',win(i)) '_' sprintf('%02d',nLen(j))]));
            y(i,j,f) = acc(expIdx);
        end
    end
end

plot(x(1,:),y(1,:,1),'-x','Color',colororder(1,:))
hold on
plot(x(1,:),y(1,:,2),'-x','Color',colororder(2,:))
plot(x(1,:),y(1,:,3),'-x','Color',colororder(3,:))
plot(x(1,:),y(1,:,4),'-x','Color',colororder(4,:))
plot(x(1,:),y(1,:,5),'-x','Color',colororder(5,:))
plot(x(1,:),y(1,:,6),'-x','Color',colororder(6,:))
plot(x(1,:),y(1,:,7),'-x','Color',colororder(7,:))
plot(x(1,:),y(1,:,8),'-x','Color',colororder(8,:))

plot(x(2,:),y(2,:,1),'-o','Color',colororder(1,:))
plot(x(2,:),y(2,:,2),'-o','Color',colororder(2,:))
plot(x(2,:),y(2,:,3),'-o','Color',colororder(3,:))
plot(x(2,:),y(2,:,4),'-o','Color',colororder(4,:))
plot(x(2,:),y(2,:,5),'-o','Color',colororder(5,:))
plot(x(2,:),y(2,:,6),'-o','Color',colororder(6,:))
plot(x(2,:),y(2,:,7),'-o','Color',colororder(7,:))
plot(x(2,:),y(2,:,8),'-o','Color',colororder(8,:))

plot(x(3,:),y(3,:,1),'-d','Color',colororder(1,:))
plot(x(3,:),y(3,:,2),'-d','Color',colororder(2,:))
plot(x(3,:),y(3,:,3),'-d','Color',colororder(3,:))
plot(x(3,:),y(3,:,4),'-d','Color',colororder(4,:))
plot(x(3,:),y(3,:,5),'-d','Color',colororder(5,:))
plot(x(3,:),y(3,:,6),'-d','Color',colororder(6,:))
plot(x(3,:),y(3,:,7),'-d','Color',colororder(7,:))
plot(x(3,:),y(3,:,8),'-d','Color',colororder(8,:))

feat_str = cell(size(feat));
for i = 1:length(feat)
    feat_str{i} = strrep(func2str(feat{i}),'_',' ');
end
legend(feat_str,'Location','southeast')

%% plot best: -mav_100_03_10000
idx = find(contains(expnames,'-mav_100_03_10000'));



