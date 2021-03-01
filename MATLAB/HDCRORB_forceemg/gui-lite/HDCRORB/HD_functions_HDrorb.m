
function message = HD_functions_HDrorb
  assignin('base','genBRandomHV', @genBRandomHV); 
  assignin('base','projBRandomHV', @projBRandomHV); 
  assignin('base','initItemMemories', @initItemMemories);
  assignin('base','projItemMemeory', @projItemMemeory); 
  assignin('base','computeNgramproj', @computeNgramproj); 
  assignin('base','hdcrorbtrain', @hdcrorbtrain); 
  assignin('base','hdcrorbpredict', @hdcrorbpredict);   
  assignin('base','genRandomHV', @genRandomHV); 
  assignin('base','downSampling', @downSampling);
  assignin('base','genTrainData', @genTrainData);
  assignin('base','lookupItemMemeory', @lookupItemMemeory);
  assignin('base','hamming', @hamming);
  assignin('base','gen_random_bipolarHV', @gen_random_bipolarHV);
  assignin('base','initbipolarItemMemories', @initbipolarItemMemories);
  assignin('base','binarize_vec',@binarize_vec);
  assignin('base','gen_HV_filler_binary',@gen_HV_filler_binary);
  message='Importing all HD functions';
end

function randomHV = gen_HV_filler_binary(D)
    randomIndex = randperm(D);
    % make half the elements 1 and the other half -1
    randomHV(randomIndex(1:floor(D/2))) = 1;
    randomHV(randomIndex(floor(D/2)+1:D)) = 0;
end


function [L_SAMPL_DATA, SAMPL_DATA] = genTrainData (data, labels, trainingFrac, order)
%
% DESCRIPTION   : generates a dataset to train the alorithm using a fraction of the input data 
%
% INPUTS:
%   data        : input data
%   labels      : input labels
%   trainingFrac: the fraction of data we should use to output a training dataset
%   order       : whether preserve the order of inputs (inorder) or randomly select
%   donwSampRate: the rate or stride of downsampling
% OUTPUTS:
%   SAMPL_DATA  : dataset for training
%   L_SAMPL_DATA: corresponding labels
%    

	rng('default');
    rng(1);
    L1 = find (labels == 1);
    L2 = find (labels == 2);
    L3 = find (labels == 3);
    L4 = find (labels == 4);
    L5 = find (labels == 5);
	L6 = find (labels == 6);
	L7 = find (labels == 7);
   
    L1 = L1 (1 : floor(length(L1) * trainingFrac));
    L2 = L2 (1 : floor(length(L2) * trainingFrac));
    L3 = L3 (1 : floor(length(L3) * trainingFrac));
    L4 = L4 (1 : floor(length(L4) * trainingFrac));
    L5 = L5 (1 : floor(length(L5) * trainingFrac));
	L6 = L6 (1 : floor(length(L6) * trainingFrac));
	L7 = L7 (1 : floor(length(L7) * trainingFrac));
 
    if order == 'inorder'
		Inx1 = 1:1:length(L1);
		Inx2 = 1:1:length(L2);
		Inx3 = 1:1:length(L3);
		Inx4 = 1:1:length(L4);
		Inx5 = 1:1:length(L5);
		Inx6 = 1:1:length(L6);
		Inx7 = 1:1:length(L7);
	else
		Inx1 = randperm (length(L1));
		Inx2 = randperm (length(L2));
		Inx3 = randperm (length(L3));
		Inx4 = randperm (length(L4));
		Inx5 = randperm (length(L5));
		Inx6 = randperm (length(L6));
		Inx7 = randperm (length(L7));
	end
    
    L_SAMPL_DATA = labels (L1(Inx1));
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L2(Inx2)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L3(Inx3)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L4(Inx4)))];
    L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L5(Inx5)))];
	L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L6(Inx6)))];
	L_SAMPL_DATA = [L_SAMPL_DATA; (labels(L7(Inx7)))];
    %L_SAMPL_DATA = L_SAMPL_DATA';
    
    SAMPL_DATA   = data (L1(Inx1), :);
    SAMPL_DATA   = [SAMPL_DATA; (data(L2(Inx2), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L3(Inx3), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L4(Inx4), :))];
    SAMPL_DATA   = [SAMPL_DATA; (data(L5(Inx5), :))];
	SAMPL_DATA   = [SAMPL_DATA; (data(L6(Inx6), :))];
	SAMPL_DATA   = [SAMPL_DATA; (data(L7(Inx7), :))];
end

function [CiMC, iMC, CiMR, iMR, CiMC_rows, CiMR_rows, iMR_rows, iMC_rows] = initItemMemories (D, MAXLC, channelsC, MAXLR, channelsR)

% DESCRIPTION   : initialize the item Memory  
% 
% INPUTS:
%   D           : Dimension of vectors
%   MAXL        : # of vectors in CiM
%   channels    : Number of acquisition channels
% OUTPUTS:
%   iM          : item memory for IDs of channels
%   CiM         : continious item memory for value of a channel
 
    % MAXL = 21;
    CiMC = containers.Map ('KeyType','double','ValueType','any');
    iMC  = containers.Map ('KeyType','double','ValueType','any');
    CiMR = containers.Map ('KeyType','double','ValueType','any');
    iMR  = containers.Map ('KeyType','double','ValueType','any');
    CiMC_rows = zeros(MAXLC+1,D);
    CiMR_rows = zeros(MAXLR+1,D);
    iMR_rows = zeros(channelsR,D);
    iMC_rows = zeros(channelsC,D);
    
    if channelsC == channelsR
        for i = 1 : channelsC
            iMC(i) = genRandomHV (D);
            iMR(i) = genRandomHV (D);
            iMC_rows(i,:) = iMC(i);
            iMR_rows(i,:) = iMR(i);
        end
    else
        for i = 1 : channelsC
            iMC(i) = genRandomHV (D);
            iMC_rows(i,:) = iMC(i);
        end

        for i = 1 : channelsR
            iMR(i) = genRandomHV (D);
            iMR_rows(i,:) = iMR(i);
        end
    end

    initHV = genRandomHV (D);
	currentHV = initHV;
	randomIndex = randperm (D);
	
    for i = 0:1:MAXLC-1
        CiMC(i) = currentHV;
        CiMC_rows(i+1,:) = CiMC(i);
        SP = floor((D/2)/MAXLC);
		startInx = (i*SP) + 1;
		endInx = ((i+1)*SP);
		currentHV (randomIndex(startInx : endInx)) = not(currentHV (randomIndex(startInx: endInx)));
    end
    
    initHV = genRandomHV (D);
	currentHV = initHV;
	randomIndex = randperm (D);
    
%     for i = 0:1:MAXLR-1
%         CiMR(i) = currentHV; 
%         CiMR_rows(i+1,:) = CiMR(i);
%         SP = floor(D/MAXLR);
% 		startInx = (i*SP) + 1;
% 		endInx = ((i+1)*SP);
% 		currentHV (randomIndex(startInx : endInx)) = not(currentHV (randomIndex(startInx: endInx)));
%     end
    for i = 0:1:MAXLR-1
        CiMR(i) = currentHV; 
        currentHV = genRandomHV(D);
    end

end
% function [CiMC, iMC, CiMR, iMR] = initItemMemories (D, MAXLC, channelsC, MAXLR, channelsR)
% %
% % DESCRIPTION    : initialize the item Memory  
% %
% % INPUTS:
% %   D            : Dimension of vectors
% %   MAXLC        : # of vectors in CiM for conditions
% %   channelsC    : Number of acquisition channels for conditions
% %   MAXLR        : # of vectors in CiM for results
% %   channelsR    : Number of acquisition channels for results
% 
% % OUTPUTS:
% %   iMC          : item memory for IDs of channels for conditions
% %   CiMC         : continious item memory for value of a channel for
% %   conditions
% %   iMR          : item memory for IDs of channels for results
% %   CiMR         : continious item memory for value of a channel for
% %   results
%  
%     % MAXL = 21;
%     CiMC = containers.Map ('KeyType','double','ValueType','any');
%     iMC  = containers.Map ('KeyType','double','ValueType','any');
%     CiMR = containers.Map ('KeyType','double','ValueType','any');
%     iMR  = containers.Map ('KeyType','double','ValueType','any');
%     rng('default');
%     rng(1);
%     
%     if channelsC == channelsR
%         for i = 1 : channelsC
%             iMC(i) = genRandomHV (D);
%             iMR(i) = genRandomHV (D);
%         end
%     else
%         for i = 1 : channelsC
%             iMC(i) = genRandomHV (D);
%         end
% 
%         for i = 1 : channelsR
%             iMR(i) = genRandomHV (D);
%         end
%     end
% 
% %     for i = 0:1:MAXL
% %         CiM(i) = currentHV; 
% %         SP = floor(D/2/MAXL);
% %       startInx = (i*SP) + 1;
% %       endInx = ((i+1)*SP) + 1;
% %       currentHV (randomIndex(startInx : endInx)) = not(currentHV (randomIndex(startInx: endInx)));
% %     end
% 
%     if MAXLC==MAXLR
%         initHVC = genRandomHV (D);
%         currentHVC = initHVC;
%         initHVR = genRandomHV (D);
%         currentHVR = initHVR;
%         for i = 0:1:MAXLC
%             CiMC(i) = currentHVC; 
%             currentHVC = genRandomHV(D);
%             CiMR(i) = currentHVR;
%             currentHVR = genRandomHV(D);
%         end
%     else
%         initHV = genRandomHV (D);
%         currentHV = initHV;
%         for i = 0:1:MAXLC
%             CiMC(i) = currentHV; 
%             currentHV = genRandomHV(D);
%         end
%         initHV = genRandomHV (D);
%         currentHV = initHV;
%         for i = 0:1:MAXLR
%             CiMR(i) = currentHV; 
%             currentHV = genRandomHV(D);
%         end
%     end
% end

function [iMC, iMR] = initbipolarItemMemories (D, channelsC, channelsR)
%
% DESCRIPTION    : initialize the item Memory  
%
% INPUTS:
%   D            : Dimension of vectors
%   channelsC    : Number of acquisition channels for conditions
%   channelsR    : Number of acquisition channels for results

% OUTPUTS:
%   iMC          : item memory for IDs of channels for conditions
%   iMR          : item memory for IDs of channels for results

 
    % MAXL = 21;
    iMC  = containers.Map ('KeyType','double','ValueType','any');
    iMR  = containers.Map ('KeyType','double','ValueType','any');
    
    if channelsC == channelsR
        for i = 1 : channelsC
            iMC(i) = gen_random_bipolarHV (D);
            iMR(i) = gen_random_bipolarHV (D);
        end
    else
        for i = 1 : channelsC
            iMC(i) = gen_random_bipolarHV (D);
        end

        for i = 1 : channelsR
            iMR(i) = gen_random_bipolarHV (D);
        end
    end

%     for i = 0:1:MAXL
%         CiM(i) = currentHV; 
%         SP = floor(D/2/MAXL);
%       startInx = (i*SP) + 1;
%       endInx = ((i+1)*SP) + 1;
%       currentHV (randomIndex(startInx : endInx)) = not(currentHV (randomIndex(startInx: endInx)));
%     end

%     if MAXLC==MAXLR
%         initHVC = genRandomHV (D);
%         currentHVC = initHVC;
%         initHVR = genRandomHV (D);
%         currentHVR = initHVR;
%         for i = 0:1:MAXLC
%             CiMC(i) = currentHVC; 
%             currentHVC = genRandomHV(D);
%             CiMR(i) = currentHVR;
%             currentHVR = genRandomHV(D);
%         end
%     else
%         initHV = genRandomHV (D);
%         currentHV = initHV;
%         for i = 0:1:MAXLC
%             CiMC(i) = currentHV; 
%             currentHV = genRandomHV(D);
%         end
%         initHV = genRandomHV (D);
%         currentHV = initHV;
%         for i = 0:1:MAXLR
%             CiMR(i) = currentHV; 
%             currentHV = genRandomHV(D);
%         end
%     end
end

function [randomHV, key] = lookupItemMemeory (itemMemory, rawKey, precision)
%
% DESCRIPTION   : recalls a vector from item Memory based on inputs
%
% INPUTS:
%   itemMemory  : item memory
%   rawKey      : the input key
%   D           : Dimension of vectors
%   precision   : precision used in quantization of input EMG signals
%
% OUTPUTS:
%   randomHV    : return the related vector

    %%quantize to integer (rounds to nearest integer, 0.5 goes up)
    key = int64 (rawKey * precision);
  
    if itemMemory.isKey (key) 
        randomHV = itemMemory (key);
    else
        fprintf ('CANNOT FIND THIS KEY: %d\n', key);       
    end
end

function randomHV = genRandomHV(D)
%
% DESCRIPTION   : generate a random vector with zero mean 
%
% INPUTS:
%   D           : Dimension of vectors
% OUTPUTS:
%   randomHV    : generated random vector

    if mod(D,2)
        disp ('Dimension is odd!!');
    else
        randomIndex = randperm (D);
        randomHV (randomIndex(1 : D/2)) = 1;
        randomHV (randomIndex(D/2+1 : D)) = 0;
       
    end
end

function randomHV = gen_random_bipolarHV(D)
    % Function Name: randomHV
    %
    % Description: Generates a random hypervector of 1's and -1's
    %
    % Arguments:
    %   D - hypervector dimension
    % 
    % Returns:
    %   randomHV - generated random hypervector
    % 
    if mod(D, 2)
        disp('Dimension is odd!!');
    else
        % generate a random vector of indices
        randomIndex = randperm(D);
        % make half the elements 1 and the other half -1
        randomHV(randomIndex(1:D/2)) = 1;
        randomHV(randomIndex(D/2+1:D)) = -1;
    end
end

function [downSampledData, downSampledLabels] = downSampling (data, labels, donwSampRate)
%
% DESCRIPTION   : apply a downsampling to get rid of redundancy in signals 
%
% INPUTS:
%   data        : input data
%   labels      : input labels
%   donwSampRate: the rate or stride of downsampling
% OUTPUTS:
%   downSampledData: downsampled data
%   downSampledLabels: downsampled labels
%    
	j = 1;
    
    for i = 1:donwSampRate:length(data(:,1))
        
		downSampledData (j,:) = data(i, :);
		downSampledLabels (j) = labels(i);
        j = j + 1;
        
    end
    
    downSampledLabels = downSampledLabels';
    
end

function [out] = binarize_vec(vec)
    out = vec;
    numzeros = sum(out==0);
    filler = gen_HV_filler_binary(numzeros);
    out(out == 0) = filler;
    out(out > 0) = 1;
    out(out < 0) = 0;
end
	
function proj_m = projBRandomHV( D, F ,q)
%   D: dim
%   F: number of features
%   q: sparsity

   proj_m=[]; 
   if mod(D,2)
        disp ('Dimension is odd!!');
   else   
    F_D=F*D;
    probM=rand(F,D);
    p_n1=(1-q)/2;
    p_p1=p_n1;

    for k=1:F
     for i=1:D
         if probM(k,i)<p_n1     
            proj_m(k,i)=-1;
         else if (p_n1<=probM(k,i)) && (probM(k,i)<(q+p_n1))
           proj_m(k,i)=0;   
         else 
           proj_m(k,i)=1; 
         end
     end
     end 
    end
   end
 
end

function randomHV = projItemMemeory (projM, voffeature,ioffeature)
%
% INPUTS:
%   projM	: random vector with {-1,0,+1}
%   voffeature	: value of a feature
%   ioffeature	: index of a feature
% OUTPUTS:
%   randomHV    : return the related vector

 
        projV=projM(ioffeature,:);
        h= voffeature.*projV;

    for i=1:length(h)
      if h(i)>0
        randomHV(i)=1;
        else
        randomHV(i)=0;
       end
    end

end

function Ngram = computeNgramproj (buffer, CiM, N, precision, iM, channels,projM)
% 	DESCRIPTION: computes the N-gram
% 	INPUTS:
% 	buffer   :  data input
% 	iM       :  Item Memory for IDs of the channels
%   N        :  dimension of the N-gram
%   precision:  precision used in quantization (no use)
% 	CiM      :  Continious Item Memory for the values of a channel (no use)
%   channels :  numeber of features
% 	OUTPUTS:
% 	Ngram    :  query hypervector
    
    %setup for first sample
    chHV = projItemMemeory (projM, buffer(1, 1),1);
    chHV = xor(chHV , iM(1));
    v = chHV;
    if channels>1    
    for i = 2 : channels
        chHV = projItemMemeory (projM, buffer(1, i), i);
        chHV = xor(chHV , iM(i));
        if i == 2
            ch2HV=chHV; 
        end
        %create matrix of all channel final outputs to be bundled
        v = [v; chHV];
    end
    
    %don't understand why adding the xor of the final and 2nd channel
    chHV = xor(chHV , ch2HV);
    v = [v; chHV]; 
    end
    
    %Add the other modalities
    if channels==1
    Ngram = v;
    else
    Ngram = mode(v);
    end
    
    %Add later samples
    for i = 2:1:N
        %replicate for other modalities
        chHV = projItemMemeory (projM, buffer(i, 1), 1);
        chHV = xor(chHV , iM(1));
        ch1HV = chHV;
        %combine the other modalities
        v = chHV;
      if channels>1  
        %replicate for other modalities
        for j = 2 : channels
            chHV = projItemMemeory (projM, buffer(i, j), j);
            chHV = xor(chHV , iM(j));
            if j == 2
                ch2HV=chHV; 
            end
            %combine the other modalities
            v = [v; chHV];
        end  
        %combine other modalities for whatever this weird thing is
        chHV = xor(chHV , ch2HV);
        v = [v; chHV]; 
      end
      
      if channels==1
        record = v;          
      else
        record = mode(v); 
      end
		Ngram = xor(circshift (Ngram, [1,1]) , record);
           
    end	 
 
end
  
function [prog_HV, result_AM, condition_AM,prog_HVlist,progHV_AM] = hdcrorbtrain (few_all, range_result, range_condition, length_training, features_condition, features_result, chAMcondition, chAMresult, iMchcondition, iMchresult, D, channels_condition, channels_result)
%
% DESCRIPTION          : train a program memory based on specific condition,result pairs
%
% INPUTS:
%   length_training    : # of training data samples
%   features_condition : condition feature training data
%   features_result    : result feature training data
%   chAMcondition      : cont. item memory for condition data
%   chAMresult         : cont. item memory for result data
%   iMchcondition      : item memory for condition channels
%   iMchresult         : item memory for result channels
%   D                  : Dimension of vectors
%   precision          : Precision determines size of CiM and also actuator quantization of returned values
%   channels_condition : number of condition channels
%   channels_results   : number of result channels
%
% OUTPUTS:
%   prog_HV            : Trained program memory
%   result_AM          : AM for result vectors
 
	result_AM = containers.Map ('KeyType','double','ValueType','any');
    condition_AM = containers.Map ('KeyType','double','ValueType','any');
    progHV_AM = containers.Map ('KeyType','double','ValueType','any');
	%known_condition = [];
    
    for i = 1:1:length_training
    	result_AM(i) = zeros (1,D);    	
    end
    for i = 1:1:length_training
    	condition_AM(i) = zeros (1,D);    	
    end
    condition_vectorlist = zeros (channels_condition, D);
    result_vectorlist = zeros (channels_result, D);
    prog_HVlist = zeros(length_training,D);
    
    i = 1;
    while i <= length_training
        for x = 1:1:channels_condition
            condition_vectorlist(x,:) = xor(chAMcondition(features_condition(i,x)),iMchcondition(x));
        end
        condition_vector_sum = condition_vectorlist;
        if channels_condition > 1
            if (mod(channels_condition, 2) == 1)
                condition_vector = mode(condition_vectorlist);
            else
                %condition_vector = mode([condition_vectorlist; circshift(condition_vectorlist(1,:), [1,1])]);
                if (few_all == 0)
                    extra_condition_vector = xor(condition_vectorlist(1,:),condition_vectorlist(2,:));
                else
                    extra_condition_vector = condition_vectorlist(1,:);
                    for y = 2:1:channels_condition
                        extra_condition_vector = xor(extra_condition_vector,condition_vectorlist(y,:));
                    end
                end
                extra_condition_vector = circshift(extra_condition_vector, [1,1]);
                condition_vector = mode([condition_vectorlist; extra_condition_vector]);
            end
        else
            condition_vector = condition_vectorlist(1,:);
        end
        for m = 1:1:channels_result
            result_vectorlist(m,:) = xor(chAMresult(features_result(i,m)),iMchresult(m));
        end
        result_vector_sum = result_vectorlist;
        %result_vector = mode(result_vectorlist);
        if channels_result > 1
            if (mod(channels_result, 2) == 1)
                result_vector = mode(result_vectorlist);
            else
                %result_vector = mode([result_vectorlist; circshift(result_vectorlist(1,:), [1,1])]);
                if (few_all == 0)
                    extra_result_vector = xor(result_vectorlist(1,:), result_vectorlist(2,:));
                else
                    extra_result_vector = result_vectorlist(1,:);
                    for y = 2:1:channels_result
                        extra_result_vector = xor(extra_result_vector,result_vectorlist(y,:));
                    end
                end
                extra_result_vector = circshift(extra_result_vector, [1,1]);
                result_vector = mode([result_vectorlist; extra_result_vector]);
            end
        else
            result_vector = result_vectorlist(1,:);
        end
        %result_vector = mode(result_vectorlist);
        %result_AM(i) = result_vector_sum;
        result_AM(i) = result_vector;
        %condition_AM(i) = condition_vector_sum;
        condition_AM(i) = condition_vector;
        protected_condition = circshift(condition_vector, [1,1]);
        prog_HVlist(i,:) = xor(protected_condition,result_vector);
        progHV_AM(i) = xor(protected_condition,result_vector);
        i = i + 1;
    end
    if (mod(i-1, 2) == 1)
        prog_HV = mode(prog_HVlist(1:i-1,:));
    else
        prog_HV = mode([prog_HVlist(1:i-1,:); genRandomHV(D)]);
    end
end

function [actuator_values, actuator_noise, condition_values, condition_noise, result_values, result_noise,progHV_values, progHV_noise] = hdcrorbpredict (progHV_AM, few_all, length_recall, prog_HV, result_AM, condition_AM, features_condition, features_result, chAMcondition, chAMresult, iMchcondition, iMchresult, D, channels_condition, channels_result)
%
% DESCRIPTION   : test accuracy based on input testing data
%
% INPUTS:
%   length_recall: # of recall samples
%   labelTestSet : testing labels
%   testSet      : EMG test data
%   AM           : Trained associative memory
%   CiM          : Cont. item memory (no use)
%   iM           : item memory
%   D            : Dimension of vectors
%   N            : size of n-gram, i.e., window size 
%   precision    : precision used in quantization (no use)
%
% OUTPUTS:
%   accuracy     : classification accuracy for all situations
%   accExcTrnz   : classification accuracy excluding the transitions between gestutes
%

    actuator_values = zeros(length_recall,1);
    actuator_noise = zeros(length_recall,1);
    condition_values = zeros(length_recall,1);
    result_values = zeros(length_recall,1);
    condition_noise = zeros(length_recall,1);
    result_noise = zeros(length_recall,1);
    progHV_values = zeros(length_recall,1);
    progHV_noise = zeros(length_recall,1);
   
    result_classes = size(result_AM);
    result_classes = result_classes(1);
    condition_classes = size(condition_AM,1);
    progHV_classes = size(progHV_AM,1);
    %[actuator_Cim_size, ~] = size(chAMresult);
    
    condition_vectorlist = zeros (channels_condition, D);
    result_vectorlist = zeros(channels_result,D);
    
    i = 1;
    while i <= length_recall
        for x = 1:1:channels_condition
            condition_vectorlist(x,:) = xor(chAMcondition(features_condition(i,x)),iMchcondition(x));
        end
        condition_vector_sum = condition_vectorlist;
        if channels_condition > 1
            if (mod(channels_condition, 2) == 1)
                condition_vector = mode(condition_vectorlist);
            else
                %condition_vector = mode([condition_vectorlist; circshift(condition_vectorlist(1,:), [1,1])]);
                if (few_all == 0)
                    extra_condition_vector = xor(condition_vectorlist(1,:),condition_vectorlist(2,:));
                else
                    extra_condition_vector = condition_vectorlist(1,:);
                    for y = 2:1:channels_condition
                        extra_condition_vector = xor(extra_condition_vector,condition_vectorlist(y,:));
                    end
                end
                extra_condition_vector = circshift(extra_condition_vector, [1,1]);
                condition_vector = mode([condition_vectorlist; extra_condition_vector]);
            end
        else
            condition_vector = condition_vectorlist(1,:);
        end
        protected_condition = circshift(condition_vector, [1,1]);
        [predict_condition, predict_cnoise] = hamming(condition_vector, condition_AM, condition_classes,0);
        condition_values(i) = predict_condition;
        condition_noise(i) = predict_cnoise;
        for m = 1:1:channels_result
            result_vectorlist(m,:) = xor(chAMresult(features_result(i,m)),iMchresult(m));
        end
        result_vector_sum = result_vectorlist;
        %result_vector = mode(result_vectorlist);
        if channels_result > 1
            if (mod(channels_result, 2) == 1)
                result_vector = mode(result_vectorlist);
            else
                %result_vector = mode([result_vectorlist; circshift(result_vectorlist(1,:), [1,1])]);
                if (few_all == 0)
                    extra_result_vector = xor(result_vectorlist(1,:), result_vectorlist(2,:));
                else
                    extra_result_vector = result_vectorlist(1,:);
                    for y = 2:1:channels_result
                        extra_result_vector = xor(extra_result_vector,result_vectorlist(y,:));
                    end
                end
                extra_result_vector = circshift(extra_result_vector, [1,1]);
                result_vector = mode([result_vectorlist; extra_result_vector]);
            end
        else
            result_vector = result_vectorlist(1,:);
        end
        [predict_result, predict_rnoise] = hamming(result_vector, result_AM, result_classes,0);
        result_values(i) = predict_result;
        result_noise(i) = predict_rnoise;
            
        prog_HV_test = xor(protected_condition,result_vector);
        [predict_prog_HV, predict_pnoise] = hamming(prog_HV_test, progHV_AM, progHV_classes,0); 
        progHV_values(i) = predict_prog_HV;
        progHV_noise(i) = predict_pnoise;
        
        noisy_resultHV = xor(protected_condition,prog_HV);
        [predict_actuator, predict_anoise] = hamming(noisy_resultHV, result_AM, result_classes,0); 
        %result_HV = result_AM(predict_result);
        actuator_values(i) = predict_actuator;
        actuator_noise(i) = predict_anoise;
%         for m = 1:1:channels_result
%             noisy_actuator_vector = xor(result_HV, iMchresult(m));
%             [predict_actuator, actuator_error] = hamming(noisy_actuator_vector, chAMresult, actuator_Cim_size,1); %#ok<ASGLU>
%             actuator_values(i,m) = predict_actuator;
%         end
        i = i + 1;
    end
end
    

function [predict_hamm, error] = hamming (q, aM, classes,iscim)
%
% DESCRIPTION       : computes the Hamming Distance and returns the prediction.
%
% INPUTS:
%   q               : query hypervector
%   AM              : Trained associative memory
%
% OUTPUTS:
%   predict_hamm    : prediction 
%

    sims = [];
    
    if (iscim)
        for j = 0 : classes-1
            sims(j+1) = sum(xor(q,aM(j)));
        end
    else
        for j = 1 : classes
            sims(j) = sum(xor(q,aM(j)));
        end
    end
    
    [error, indx]=min(sims');
    
    if (iscim)
        predict_hamm=indx-1;
    else
        predict_hamm = indx;
    end
     
end
