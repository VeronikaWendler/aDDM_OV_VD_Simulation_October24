% Adapted from Ting & Gluth (2024)
%
%1. increase the range of OV and VD
%2. use efficient way to sample
%3. update the way to save data (for each virtual subject)
%
% Simulation of aDDM to account for magmitude effect
% - In this simple simulation, five parameters will be manipulated
%
% 1. Drift rate
% 2. Va and Vb
% 3. SD of noise
% 4. threshold
% 5. Probability of looking at option A
% 6. theta: Impact of attention
% - Simulate 10-70 subjects given range of parameters. 4 conditions are:
% created:
% 1. Va = 3, Vb = 1 (OV = 4, VD =2)
% 2. Va = 6, Vb = 2 (OV = 8, VD =4)
% 3. Va = 8, Vb = 4 (OV = 12, VD =4)
% 4. Va = 9, Vb = 7 (OV = 16, VD =2)

%initialize the environment
clear; clc
rng('shuffle')
% Basic aDDM
% parameters
n_sim   = 200; 
subjset = 10:10:60;                                                         %specifies a set of subjects, ranging from 10 to 70 in steps of 10.
NsubjSet= length(subjset); 
trialset = 20:10:60;                                                        %sets the number of trials, ranging from 20 to 60 in steps of 10. but we only focus on 20-50 in the analysis
NtrialSet=length(trialset);

% aDDM parameters. The first three ranges of parameters were selected based on the Smith and Krajbich,
% 2019 (see simulateADDM_JMP.R)
minTheta = 0.2;
maxTheta = 0.4;
minNDT = 350;
maxNDT = 650;
minSigma = 0.02;
maxSigma = 0.03;
minD = 0.0002;
maxD = 0.0004;
threshold = [1 -1];
% option values
values = [3 1; 6 2; 8 4; 9 7];
valuesALL = cell(1,NtrialSet);
for k_trialset = 1:NtrialSet
    nTrial = trialset(k_trialset);
    valuesALL{k_trialset} = repmat([values(:,1) values(:,2)],nTrial,1);
end

% create Derivatives\Results_2 folder if not already existing
outputDir = 'Derivatives\Results_2';

if ~exist(outputDir, 'dir')
    mkdir(outputDir);
end

% start simulations
for k_sim = 25:n_sim
    AllData = cell(NsubjSet,NtrialSet); % predetermine the size of data structure
    
    for k_trialset = 1:NtrialSet
        totalTrial = trialset(k_trialset)*4; % each cell repeats n times. n is drawen from trialset
        
        for k_subjSet = 1:NsubjSet
            nSubj = subjset(k_subjSet);
            params = nan(nSubj,4);
            params(:,1) = normrnd(minTheta+rand*(maxTheta-minTheta),.01,nSubj,1); % theta
            params(:,2) = normrnd(minSigma+rand*(maxSigma-minSigma),.001,nSubj,1); % SD_noise
            params(:,3) = normrnd(minD+rand*(maxD-minD),.00001,nSubj,1); % drift
            params(:,4) = normrnd(minNDT+rand*(maxNDT-minNDT),10,nSubj,1); % NDT
            data   = nan(nSubj*totalTrial,15);  % predetermine the size of data structure. total 15 variables
            
            for k_sub = 1:nSubj
                VaList = valuesALL{k_trialset}(:,1)+rand(totalTrial,1)./3; % Better option + random numbers
                VbList = valuesALL{k_trialset}(:,2)-rand(totalTrial,1)./3; % Worse option - random numbers, so that the value difference is not a certain number
                
                for k_trial = 1:totalTrial
                    [Choice, RT, E, gazeA, gazeB, nSwitch]=EvidenceAccumulate_v2(VaList(k_trial),VbList(k_trial),params(k_sub,:),threshold);
                    data((k_sub-1)*totalTrial+k_trial,:) = [k_sub+k_trialset*1000 k_trial VaList(k_trial) VbList(k_trial) Choice RT gazeA gazeB nSwitch params(k_sub,:) threshold];
                    % subjID, trial# BetterOption(A) WorseOption(B), Accuracy, RT, gazeA(ms) gazeB(ms), #switch, six parameters 
                end
            end
            AllData{k_subjSet,k_trialset} = data;
        end
    end
    save(fullfile(outputDir, ['aDDM_stimulation' num2str(k_sim)]), 'AllData');
end
