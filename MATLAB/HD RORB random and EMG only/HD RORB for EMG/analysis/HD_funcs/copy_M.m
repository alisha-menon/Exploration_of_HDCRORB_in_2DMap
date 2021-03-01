function [newM] = copy_M(oldM)
    newM = containers.Map('KeyType','int32','ValueType','any');
    classes = oldM.keys;
    for i = 1:1:size(classes, 2)
        newM(cell2mat(classes(i))) = oldM(cell2mat(classes(i)));
    end
end