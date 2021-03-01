function rotated_func(model, gestures, trainFeatures, testFeatures, Nshot, Nshift, outfname)
    
    numTrials = size(trainFeatures,2);
    internal_model = shift_array(model,Nshift);
    internal_model = reset_AM(internal_model, numTrials, gestures);
    internal_model = train_model(internal_model, trainFeatures);
    
    info.gestures = gestures;
    info.trainFeatures = trainFeatures;
    info.testFeatures = testFeatures;
    info.model = internal_model;
    info.Nshift = Nshift;
    % replace eM with eM_rot
    internal_model.eM = internal_model.eM_rot;
    
    for n = Nshot
        info.Nshot = n;
        out = test_new_context(internal_model,testFeatures,n);
        outfnameN = [outfname '-' num2str(n) '-' num2str(n)];
        save_stats(out, info, outfnameN)
    end
end