function [experiments] = load_subject_data(subject)
    load('./info/info.mat','subjectInfo','ex','gestList','dataDir');
    dates = subjectInfo(subject).dates;
    
    experiments = cell(1,length(ex));
    
    for k = 1:length(ex)
        e = ex(k);
        recDate = dates{k};
        experiments{k} = struct([]);
        gest = gestList{k};
        for i = 1:length(gest)
            d = dir([dataDir num2str(subject,'%03.f') '/' num2str(e) '/' recDate '*/*' num2str(gest(i),'%03.f') '_' recDate '*']);
            files = {d.name};
            folders = {d.folder};
            % load all files associated with this gesture
            for j = 1:length(files)
                load([folders{j} '/' files{j}]);
                experiments{k}(i,j).raw = double(data(:,1:64));
                experiments{k}(i,j).label = double(gestLabel);
                experiments{k}(i,j).info = streamInfo;
                clearvars data gestLabel streamInfo
            end
        end
    end
    
    if subjectInfo(subject).extraEffort
        experiments{4} = experiments{4}(:,2:6);
        experiments{5} = experiments{5}(:,2:6);
        experiments{6} = experiments{6}(:,2:6);
    end
end