load('channel_scores_alisha.mat')
%64 channels, 5 subjects, 1 is fisher score, the second is tree-based
%feature importance
function [] = plot_emg_array(z,size,range)
    % right half first
    x = zeros(64,1);
    y = zeros(64,1);
    cSpace = 1.24;
    rSpace = 1;
    offset = rSpace/2;
    for i = 1:32
        x(i) = floor((i-1)/4)*cSpace + cSpace/2;
        y(i) = mod(i-1,4)*rSpace;
        if mod(floor((i-1)/4),2)
            y(i) = y(i) + offset;
        end
    end
    for i = 1:32
        x(i+32) = floor((i-1)/4)*cSpace + cSpace/2 - 8*cSpace;
        y(i+32) = (3-mod(i-1,4))*rSpace;
        if mod(floor((i-1)/4),2)
            y(i+32) = y(i+32) + offset;
        end
    end
    scatter(x,y,size,z,'filled');
%     text(x,y,strsplit(num2str(z)),'HorizontalAlignment','center','VerticalAlignment','middle','Color','white')
    set(gca,'clim',range)
    axis equal
    axis off
end