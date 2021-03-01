% Function Name: program_memory
%
% Description: creates a program memory from representative condition and
% result samples for each gesture
%
% Arguments:
%   model for result and model for condition vectors, and the AMs for each
%   that contain the representative vectors per gesture
% 
% Returns:
%   program vector
%

function [program_vec, distance_list, prog_HVlist] = program_memory(model_r, model_c, AM_r, AM_c)
    classes = AM_r.keys;
    prog_HVlist = zeros(size(classes, 2),model_r.D);
    distance_list = zeros(size(classes, 2), 1);
    for i = 1:1:size(classes, 2)
        temp_r = AM_r(cell2mat(classes(i)));
        temp_c = AM_c(cell2mat(classes(i)));
        protected_condition = circshift(temp_c, [1,1]);  
        prog_HVlist(i,:) = xor(temp_r,protected_condition);
    end
    if (mod(size(classes,2), 2) == 1)
        program_vec = mode(prog_HVlist(1:size(classes,2),:));
    else
        program_vec = mode([prog_HVlist(1:size(classes,2),:); gen_random_HV_bin(model_c.D)]);
    end
    for i = 1:1:size(classes, 2)
        distance_list(i) = hamming_distance(prog_HVlist(i,:),program_vec);
    end
end