close all
clear
clc

files = dir('./*.mat');

for f = 1:length(files)
    load(files(f).name);
    expName = files(f).name(1:end-4);
    for s = 1:length(experiment)
        sessName = experiment(s).description;
        zoe = load(['../chgpts/realtime_experiments_breakpoints/' expName '_s' num2str(s) '.mat']);
        for g = 1:length(experiment(s).trials)
            % tkeo chgpts
            load(['../chgpts/' expName '_' sessName '_' num2str(g-1) '.mat']);
            experiment(s).trials(g).onsetTKEOMean = round(mean(onsets(~isoutlier(onsets(:,1)),1))/50);
            experiment(s).trials(g).offsetTKEOMean = round(mean(offsets(~isoutlier(offsets(:,1)),1))/50);
            experiment(s).trials(g).onsetTKEOMed = round(median(onsets(~isoutlier(onsets(:,1)),1))/50);
            experiment(s).trials(g).offsetTKEOMed = round(median(offsets(~isoutlier(offsets(:,1)),1))/50);
            % matlab chgpts
            onset = find(diff(experiment(s).trials(g).offlineLabel)>0);
            if ~isempty(onset)
                experiment(s).trials(g).onsetMat = onset;
            else
                experiment(s).trials(g).onsetMat = 1;
            end
            offset = find(diff(experiment(s).trials(g).offlineLabel)<0);
            if ~isempty(offset)
                experiment(s).trials(g).offsetMat = offset;
            else
                experiment(s).trials(g).offsetMat = length(experiment(s).trials(g).offlineLabel);
            end
            % zoe no tkeo
            onsets = zoe.onset(g,:);
            experiment(s).trials(g).onsetZoeMean = round(mean(onsets(~isoutlier(onsets)))/50);
            experiment(s).trials(g).onsetZoeMed = round(median(onsets(~isoutlier(onsets)))/50);
            offsets = zoe.offset(g,:);
            experiment(s).trials(g).offsetZoeMean = round(mean(offsets(~isoutlier(offsets)))/50);
            experiment(s).trials(g).offsetZoeMed = round(median(offsets(~isoutlier(offsets)))/50);
        end
    end
    save(files(f).name,'experiment');
end

function plot_timing(trial)
    ax1 = subplot(4,1,1);
    tk = tkeo_emg(trial.raw);
    plot(tk);
    xlim([1 length(trial.raw)])
    
    % matlab onset/offset
    onset = find(diff(trial.offlineLabel)>0);
    offset = find(diff(trial.offlineLabel)<0);
    
    ax2 = subplot(4,1,2);
    if ~isempty(onset)
        patch(([onset onset+1 onset+1 onset] - 1).*50 + 1,[0 0 1 1],[0.8 0.8 0.8])
        hold on
    end
    if ~isempty(offset)
        patch(([offset offset+1 offset+1 offset] - 1).*50 + 1,[0 0 1 1],[0.8 0.8 0.8])
        hold on
    end
    xline(round(median(trial.onsets(:,1))/50)*50+25,'g','LineWidth',2);
    hold on
    xline(round(median(trial.offsets(:,1))/50)*50+25,'g','LineWidth',2);
    histogram(trial.onsets(:,1),1:50:length(trial.raw),'Normalization','probability')
    histogram(trial.offsets(:,1),1:50:length(trial.raw),'Normalization','probability')
    xlim([1 length(trial.raw)])
    
    ax3 = subplot(4,1,3);
    imagesc([1 length(trial.raw)], [1 64], trial.features')
    xlim([1 length(trial.raw)])
    
    ax4 = subplot(4,1,4);
    pred = trial.prediction';
    pred = repmat(pred,50,1);
    pred = reshape(pred,numel(pred),1);
    plot(pred)
    xlim([1 length(trial.raw)])
    
    linkaxes([ax1 ax2 ax3 ax4],'x');
end

function proc = tkeo_emg(data)
    data = highpass(data,0.1,500);
    a = data.^2;
    b = [data(1,:); data(1:end-1,:)];
    c = [data(2:end,:); data(end,:)];
    proc = abs(a - (b.*c));
    proc = lowpass(proc,5,1000);
end