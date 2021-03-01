function [c,f] = generate_confusion_fig(titleTxt, actualGest, predictedGest, gestures)
    % create gesture dictionary
    gestDict = containers.Map ('KeyType','int32','ValueType','any');
    gestDict(100) = 'Rest';
    gestDict(101) = 'Index Flexion';
    gestDict(102) = 'Index Extension';
    gestDict(103) = 'Middle Flexion';
    gestDict(104) = 'Middle Extension';
    gestDict(105) = 'Ring Flexion';
    gestDict(106) = 'Ring Extension';
    gestDict(107) = 'Pinky Flexion';
    gestDict(108) = 'Pinky Extension';
    gestDict(109) = 'Thumb Flexion';
    gestDict(110) = 'Thumb Extension';
    gestDict(111) = 'Thumb Adduction';
    gestDict(112) = 'Thumb Abduction';
    gestDict(201) = 'One';
    gestDict(202) = 'Two';
    gestDict(203) = 'Three';
    gestDict(204) = 'Four';
    gestDict(205) = 'Five';
    gestDict(206) = 'Thumb Up';
    gestDict(207) = 'Fist';
    gestDict(208) = 'Flat';
    
    gestures = sort(gestures);
    c = confusionmat(actualGest,predictedGest,'Order',gestures);
    totalTrials = mean(sum(c,2));
    c = c./totalTrials.*100;
    c = round(c.*100)./100;
    
    gestureLabels = cell(size(gestures));
    for i = 1:length(gestures)
        gestureLabels{i} = gestDict(gestures(i));
    end
    
    baseSize = 700;
    baseGest = 13;
    scaledSize = round(baseSize*length(gestures)/baseGest);
    
    f = figure;
    set(f,'Position',[100 100 scaledSize 2*scaledSize])
    imagesc(c)
    colormap(flipud(gray(2048)))
    axis square
    xticks(1:length(gestures));
    xticklabels(gestureLabels);
    xtickangle(45);
    yticks(1:length(gestures));
    yticklabels(gestureLabels);
    set(gca, 'FontSize', 14)
    
    ylabel('Actual Gesture','FontSize',16,'FontWeight','bold')
    xlabel('Predicted Gesture','FontSize',16,'FontWeight','bold')
    title(titleTxt,'FontSize',24,'FontWeight','bold')
    colorbar
    dx = 0.3;
    for i = 1:length(gestures)
        for j = 1:length(gestures)
            if c(i,j) > 0.01
                accTxt = num2str(c(i,j));
                dxScaled = dx*length(accTxt)/5;
                if i == j
                    text(j-dxScaled-0.1,i, accTxt,'Color','white','FontSize',12)
                else
                    text(j-dxScaled-0.1,i, accTxt,'Color','red','FontSize',12)
                end
            end
        end
    end
end