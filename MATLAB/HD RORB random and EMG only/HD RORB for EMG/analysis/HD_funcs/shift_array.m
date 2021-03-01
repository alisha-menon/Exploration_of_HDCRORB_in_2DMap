function [model] = shift_array(model, Nshift)

    map1 = 1:32;
    map1 = reshape(map1, [4,8]);
    map1 = flipud(map1);
    map2 = 33:64;
    map2 = reshape(map2, [4,8]);
    map = [map2, map1];
    map1 = [];
    map2 = map;

    for c = 1:size(map,2)
        if all(model.eM(map(1,c)) == 0)
            map1 = [map1,map(:,c)];
            map2 = map2(:,2:end);
        end
    end

    map2 = circshift(map2,Nshift,2);
    map_rot = [map1 map2];
    model.eM_rot = containers.Map ('KeyType','int32','ValueType','any');

    for i=1:model.noCh
        model.eM_rot(map_rot(i)) = model.eM(map(i));
    end

end