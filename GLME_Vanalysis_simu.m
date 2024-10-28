clear all;
clc;
k_trial = 5;
folderName = [pwd '/Derivatives/Results_2'];
listing = dir(fullfile(folderName, '*.mat'));
simFiles = {listing(3:end).name};
nSim = 200;

glmeResultsFolder = [pwd '/Derivatives/glmeResults_noRand'];  
ensembleResultsFolder = [pwd '/Derivatives/ensembleResults'];

% Check GLME results folder 
if ~exist(glmeResultsFolder, 'dir')
    mkdir(glmeResultsFolder);
end

% Check Ensemble results folder 
if ~exist(ensembleResultsFolder, 'dir')
    mkdir(ensembleResultsFolder);
end

for k_sim = 1:nSim
    
    load([folderName '/' simFiles{k_sim}])
    nDataset    = length(AllData);
    
    str = simFiles{k_sim}; % First define the string.
    str1 = str(17:end-4); % Gives a character
    dbl1 = str2double(str1)  % Give a number (double)

    %% Analyze
    for k_subjSet = 1:nDataset
        clear glme tbl
        tblALL = AllData{k_subjSet,k_trial};
        V1 = tblALL(:,3);                    % value of the higher
        V2 = tblALL(:,4);                    % value of the lower option
        OV = V1 + V2;
        VD = V1 - V2;
        absVD = abs(VD);
        subj = tblALL(:,1);
        RT = tblALL(:,6);
        logRT = log(tblALL(:,6));
        Accuracy = logical(tblALL(:,5));

        Choice = tblALL(:,5);  % 1 if higher option chosen, 0 if lower option chosen
        DwellHigher = tblALL(:,7);  % Time spent on higher option
        DwellLower = tblALL(:,8);   % Time spent on lower option
        
        % Dwellopt initialization
        Dwellopt = zeros(size(Choice));
        
        % Assign Dwellopt based on the chosen option
        for i = 1:length(Choice)
            if Choice(i) == 1        
                Dwellopt(i) = DwellHigher(i);
            else  
                Dwellopt(i) = DwellLower(i);
            end
        end
    
        tbl = table(subj,V1,V2,Accuracy,RT,logRT,OV,VD,absVD,DwellHigher,DwellLower,Dwellopt);
   
        if any(ismissing(tbl), 'all') || any(isinf(tbl.V1(:))) || any(isinf(tbl.V2(:))) || ...
                any(isinf(tbl.OV(:))) || any(isinf(tbl.absVD(:))) || any(isinf(tbl.Dwellopt(:))) || ...
                any(isinf(tbl.RT(:))) || any(isinf(tbl.DwellHigher(:))) || any(isinf(tbl.DwellLower(:)))
                 warning('NaN or Inf values detected in the dataset. Skipping this dataset.');
                 continue;
        end


        % GLME with full random effects
        glme.CORR = fitglme(tbl,'Accuracy ~ 1 + OV + absVD + (1 | subj)','Distribution','Binomial');
        glme.RT = fitglme(tbl,'RT ~ 1 + OV + absVD+ (1 | subj)','Distribution','Normal');
        glme.Dwell = fitglme(tbl,'Dwellopt ~ 1 + OV + absVD + (1  | subj)', 'Distribution', 'Normal');
        glme.CORR_adv = fitglme(tbl,'Accuracy ~ 1 + OV + absVD + Dwellopt + (1 | subj)','Distribution','Binomial');
        glme.Dwell_rt = fitglme(tbl,'Dwellopt ~ 1 + OV + absVD + RT + (1 | subj)', 'Distribution', 'Normal');
        glme.RT_dwell = fitglme(tbl,'RT ~ 1 + OV + absVD + Dwellopt + (1 | subj)', 'Distribution', 'Normal');
        glme.CORR_dwellbo = fitglme(tbl,'Accuracy ~ 1 + DwellHigher + absVD + DwellLower + (1 | subj)', 'Distribution', 'Binomial');

        % non linear functions 
        glme.Non_Lin_CORR = fitglme(tbl, 'Accuracy ~ 1 + OV^2 + absVD^2 + (1 | subj)', 'Distribution', 'Binomial');

        % Trees
        ensemble.CORR = fitensemble(tbl, 'Accuracy ~ 1 + OV + absVD', 'AdaBoostM1', 100, 'Tree');  % Boosted trees for classification
        ensemble.RT = fitensemble(tbl, 'RT ~ 1 + OV + absVD', 'LSBoost', 100, 'Tree');  % Boosted trees for classification
        ensemble.Dwell = fitensemble(tbl, 'Dwellopt ~ 1 + OV + absVD', 'LSBoost', 100, 'Tree');  % Boosted trees for classification

        save(fullfile(glmeResultsFolder, ['glme_sim' str1 '_subjN' num2str(k_subjSet) '_trial' num2str(k_trial)]), 'glme', 'tbl');
        save(fullfile(ensembleResultsFolder, ['ensemble_sim' str1 '_subjN' num2str(k_subjSet) '_trial' num2str(k_trial)]), 'ensemble', 'tbl');
        
    end
end

%% load data
clear all;
clc;
k_trial = 5;
nSim = 200;

folderName = [pwd '/Derivatives/Results_summary_GlmeEns'];
if ~exist(folderName, 'dir')
    mkdir(folderName);
end
listing = dir(fullfile(folderName, '*.mat'));
simFiles = {listing(3:end).name};

% GLME and Ensemble results folders
glmeResultsFolder = [pwd '/Derivatives/glmeResults_noRand'];
ensembleResultsFolder = [pwd '/Derivatives/ensembleResults'];

if ~exist(glmeResultsFolder, 'dir')
    mkdir(glmeResultsFolder);
end

if ~exist(ensembleResultsFolder, 'dir')
    mkdir(ensembleResultsFolder);
end

% Get GLME result files
glmelistingt = dir([glmeResultsFolder '/*trial' num2str(k_trial) '.mat']);
SuccSimList = length(glmelistingt);
glmeList = {glmelistingt.name};

% Get Ensemble result files
ensemblelistingt = dir([ensembleResultsFolder '/*trial' num2str(k_trial) '.mat']);
ensembleList = {ensemblelistingt.name};

nDataset = 6;

% Separate count for GLME and Ensemble models
count_glme = zeros(1, nDataset + 1);
count_ensemble = zeros(1, nDataset + 1);

% Initialize GLME and Ensemble structures
SUMMARY.glme = struct();
SUMMARY.ensemble = struct();

% Initialize GLME fields
SUMMARY.glme.corrP = nan(nSim, nDataset, 3);
SUMMARY.glme.rtP = nan(nSim, nDataset, 3);
SUMMARY.glme.DwellP = nan(nSim, nDataset, 3);
SUMMARY.glme.corr_advP = nan(nSim, nDataset, 4);
SUMMARY.glme.Dwell_rtP = nan(nSim, nDataset, 4);
SUMMARY.glme.RT_dwellP = nan(nSim, nDataset, 4);
SUMMARY.glme.corr_dwellboP = nan(nSim, nDataset, 4);
SUMMARY.glme.non_Lin_corrP = nan(nSim, nDataset, 5);

SUMMARY.glme.corrBeta = nan(nSim, nDataset, 3);
SUMMARY.glme.rtBeta = nan(nSim, nDataset, 3);
SUMMARY.glme.DwellBeta = nan(nSim, nDataset, 3);
SUMMARY.glme.corr_advBeta = nan(nSim, nDataset, 4);
SUMMARY.glme.Dwell_rtBeta = nan(nSim, nDataset, 4);
SUMMARY.glme.RT_dwellBeta = nan(nSim, nDataset, 4);
SUMMARY.glme.corr_dwellboBeta = nan(nSim, nDataset, 4);
SUMMARY.glme.non_Lin_corrBeta = nan(nSim, nDataset, 5);

% Initialize Ensemble fields
SUMMARY.ensemble.CORR_Accuracy = nan(nSim, nDataset);
SUMMARY.ensemble.RT_Error = nan(nSim, nDataset);
SUMMARY.ensemble.Dwell_Error = nan(nSim, nDataset);

missingGLME = {};
missingEnsemble = {};

% Loop through simulations
for k_sim = 1:200
    for k_subjSet = 1:nDataset
        currPro = SuccSimList - sum(count_glme);  % Progress tracking
        resultNAME = ['glme_sim' num2str(k_sim) '_subjN' num2str(k_subjSet) '_trial' num2str(k_trial) '.mat'];

        % Load GLME results
        if isfile([pwd '/Derivatives/glmeResults_noRand/' resultNAME])
            load([pwd '/Derivatives/glmeResults_noRand/' resultNAME]);
            count_glme(k_subjSet) = count_glme(k_subjSet) + 1;

            % Store p-values and coefficients
            SUMMARY.glme.corrP(k_sim, k_subjSet, :) = glme.CORR.Coefficients.pValue;
            SUMMARY.glme.rtP(k_sim, k_subjSet, :) = glme.RT.Coefficients.pValue;
            SUMMARY.glme.DwellP(k_sim, k_subjSet, :) = glme.Dwell.Coefficients.pValue;
            SUMMARY.glme.corr_advP(k_sim, k_subjSet, :) = glme.CORR_adv.Coefficients.pValue;
            SUMMARY.glme.Dwell_rtP(k_sim, k_subjSet, :) = glme.Dwell_rt.Coefficients.pValue;
            SUMMARY.glme.RT_dwellP(k_sim, k_subjSet, :) = glme.RT_dwell.Coefficients.pValue;
            SUMMARY.glme.corr_dwellboP(k_sim, k_subjSet, :) = glme.CORR_dwellbo.Coefficients.pValue;
            SUMMARY.glme.non_Lin_corrP(k_sim, k_subjSet, :) = glme.Non_Lin_CORR.Coefficients.pValue;

            SUMMARY.glme.corrBeta(k_sim, k_subjSet, :) = glme.CORR.Coefficients.Estimate;
            SUMMARY.glme.rtBeta(k_sim, k_subjSet, :) = glme.RT.Coefficients.Estimate;
            SUMMARY.glme.DwellBeta(k_sim, k_subjSet, :) = glme.Dwell.Coefficients.Estimate;
            SUMMARY.glme.corr_advBeta(k_sim, k_subjSet, :) = glme.CORR_adv.Coefficients.Estimate;
            SUMMARY.glme.Dwell_rtBeta(k_sim, k_subjSet, :) = glme.Dwell_rt.Coefficients.Estimate;
            SUMMARY.glme.RT_dwellBeta(k_sim, k_subjSet, :) = glme.RT_dwell.Coefficients.Estimate;
            SUMMARY.glme.corr_dwellboBeta(k_sim, k_subjSet, :) = glme.CORR_dwellbo.Coefficients.Estimate;
            SUMMARY.glme.non_Lin_corrBeta(k_sim, k_subjSet, :) = glme.Non_Lin_CORR.Coefficients.Estimate;

        else
            count_glme(7) = count_glme(7) + 1;
            missingGLME{count_glme(7)} = resultNAME;
        end

        % Load Ensemble results
        ensembleResultName = ['ensemble_sim' num2str(k_sim) '_subjN' num2str(k_subjSet) '_trial' num2str(k_trial) '.mat'];
        if isfile(fullfile(ensembleResultsFolder, ensembleResultName))
            load(fullfile(ensembleResultsFolder, ensembleResultName));
            count_ensemble(k_subjSet) = count_ensemble(k_subjSet) + 1;

            % Calculate classification accuracy for AdaBoostM1 model
            SUMMARY.ensemble.CORR_Accuracy(k_sim, k_subjSet) = 1 - loss(ensemble.CORR, tbl, 'LossFun', 'classiferror');
            % Calculate regression error (MSE) for RT prediction
            SUMMARY.ensemble.RT_Error(k_sim, k_subjSet) = loss(ensemble.RT, tbl, 'LossFun', 'mse');
            % For Dwell, if it's also regression
            SUMMARY.ensemble.Dwell_Error(k_sim, k_subjSet) = loss(ensemble.Dwell, tbl, 'LossFun', 'mse');

        else
            count_ensemble(7) = count_ensemble(7) + 1;
            missingEnsemble{count_ensemble(7)} = ensembleResultName;
        end
    end
end

% Convert missing files to cell arrays
missingGLME = missingGLME';
missingEnsemble = missingEnsemble';

overviewBeta.OVonCORR = nanmean(squeeze(SUMMARY.glme.corrBeta(:, :, 2)));
overviewBeta.OVonRT= nanmean(squeeze(SUMMARY.glme.rtBeta(:,:,2)));
overviewBeta.OVonDwell= nanmean(squeeze(SUMMARY.glme.DwellBeta(:,:,2)));
overviewBeta.OVonCORRadv= nanmean(squeeze(SUMMARY.glme.corr_advBeta(:,:,2)));
overviewBeta.OVonDwell_rt = nanmean(squeeze(SUMMARY.glme.Dwell_rtBeta(:,:,2)));
overviewBeta.OVonRT_dwell = nanmean(squeeze(SUMMARY.glme.RT_dwellBeta(:,:,2)));
overviewBeta.OVonCORR_dwellbo = nanmean(squeeze(SUMMARY.glme.corr_dwellboBeta(:,:,2)));
overviewBeta.OVonCORRNon_Lin_CORR = nanmean(squeeze(SUMMARY.glme.non_Lin_corrBeta(:,:,4)));

overviewBeta.VDonCORR= nanmean(squeeze(SUMMARY.glme.corrBeta(:,:,3)));
overviewBeta.VDonRT= nanmean(squeeze(SUMMARY.glme.rtBeta(:,:,3)));
overviewBeta.VDonDwell= nanmean(squeeze(SUMMARY.glme.DwellBeta(:,:,3)));
overviewBeta.VDonCORRadv= nanmean(squeeze(SUMMARY.glme.corr_advBeta(:,:,3)));
overviewBeta.VDonDwell_rt = nanmean(squeeze(SUMMARY.glme.Dwell_rtBeta(:,:,3)));
overviewBeta.VDonRT_dwell = nanmean(squeeze(SUMMARY.glme.RT_dwellBeta(:,:,3)));
overviewBeta.VDonCORR_dwellbo = nanmean(squeeze(SUMMARY.glme.corr_dwellboBeta(:,:,3)));
overviewBeta.VDonCORRNon_Lin_CORR = nanmean(squeeze(SUMMARY.glme.non_Lin_corrBeta(:,:,5)));



overviewP.OVonCORR= nansum(squeeze(SUMMARY.glme.corrP(:,:,2))<0.05);
overviewP.OVonRT= nansum(squeeze(SUMMARY.glme.rtP(:,:,2))<0.05);
overviewP.OVonDwell= nansum(squeeze(SUMMARY.glme.DwellP(:,:,2))<0.05);
overviewP.OVonCORRadv= nansum(squeeze(SUMMARY.glme.corr_advP(:,:,2))<0.05);
overviewP.VDonCORR= nansum(squeeze(SUMMARY.glme.corrP(:,:,3))<0.05);
overviewP.VDonRT= nansum(squeeze(SUMMARY.glme.rtP(:,:,3))<0.05);
overviewP.VDonDwell= nansum(squeeze(SUMMARY.glme.DwellP(:,:,3))<0.05);
overviewP.VDonCORRadv= nansum(squeeze(SUMMARY.glme.corr_advP(:,:,3))<0.05);
overviewP.OVonDwell_rt = nansum(squeeze(SUMMARY.glme.Dwell_rtP(:,:,2))<0.05);
overviewP.OVonRT_dwell = nansum(squeeze(SUMMARY.glme.RT_dwellP(:,:,2))<0.05);
overviewP.OVonCORR_dwellbo = nansum(squeeze(SUMMARY.glme.corr_dwellboP(:,:,2))<0.05);
overviewP.OVonCORRNon_Lin_CORR = nanmean(squeeze(SUMMARY.glme.non_Lin_corrP(:,:,4))<0.05);

overviewP.VDonDwell_rt = nansum(squeeze(SUMMARY.glme.Dwell_rtP(:,:,3))<0.05);
overviewP.VDonRT_dwell = nansum(squeeze(SUMMARY.glme.RT_dwellP(:,:,3))<0.05);
overviewP.VDonCORR_dwellbo = nansum(squeeze(SUMMARY.glme.corr_dwellboP(:,:,3))<0.05);
overviewP.VDonCORRNon_Lin_CORR = nanmean(squeeze(SUMMARY.glme.non_Lin_corrP(:,:,5))<0.05);


estimatedP.OVonCORR= overviewP.OVonCORR./count_glme(1:6);
estimatedP.OVonRT= overviewP.OVonRT./count_glme(1:6);
estimatedP.OVonDwell= overviewP.OVonDwell./count_glme(1:6);
estimatedP.OVonCORRadv= overviewP.OVonCORRadv./count_glme(1:6);
estimatedP.VDonCORR= overviewP.VDonCORR./count_glme(1:6);
estimatedP.VDonRT= overviewP.VDonRT./count_glme(1:6);
estimatedP.VDonDwell= overviewP.VDonDwell./count_glme(1:6);
estimatedP.VDonCORRadv= overviewP.VDonCORRadv./count_glme(1:6);
estimatedP.OVonDwell_rt = overviewP.OVonDwell_rt./count_glme(1:6);
estimatedP.OVonRT_dwell = overviewP.OVonRT_dwell./count_glme(1:6);
estimatedP.OVonCORR_dwellbo = overviewP.OVonCORR_dwellbo./count_glme(1:6);
estimatedP.OVonCORRNon_Lin_CORR = overviewP.OVonCORRNon_Lin_CORR./count_glme(1:6);
estimatedP.VDonDwell_rt = overviewP.VDonDwell_rt./count_glme(1:6);
estimatedP.VDonRT_dwell = overviewP.VDonRT_dwell./count_glme(1:6);
estimatedP.VDonCORR_dwellbo = overviewP.VDonCORR_dwellbo./count_glme(1:6);
estimatedP.VDonCORRNon_Lin_CORR = overviewP.VDonCORRNon_Lin_CORR./count_glme(1:6);



overviewEnsemble.CORR_Accuracy = nanmean(SUMMARY.ensemble.CORR_Accuracy, 1);
overviewEnsemble.RT_Error = nanmean(SUMMARY.ensemble.RT_Error, 1);
overviewEnsemble.Dwell_Error = nanmean(SUMMARY.ensemble.Dwell_Error, 1);

% save(fullfile(folderName, ['ResultSummary_GLME_20210901_trial_' num2str(k_trial)]), ...
%     'SUMMARY', 'overviewP', 'overviewBeta', 'missingGLME', 'count', 'estimatedP');
% Save results for both GLME and Ensemble separately
save(fullfile(folderName, ['ResultSummary_GLME_' num2str(k_trial)]), 'SUMMARY', 'overviewBeta','overviewP', 'missingGLME', 'count_glme', 'estimatedP');
save(fullfile(folderName, ['ResultSummary_Ensemble_' num2str(k_trial)]), 'SUMMARY', 'overviewEnsemble', 'missingEnsemble', 'count_ensemble');