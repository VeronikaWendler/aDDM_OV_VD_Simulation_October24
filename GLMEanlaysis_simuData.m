% analyze data with GLME
clear all
clc
k_trial = 4;
folderName = [pwd '/simulation_20210824'];
listing = dir(folderName);
simFiles= {listing(3:end).name};
nSim = 100;
for k_sim = 1:nSim
    
    load([folderName '/' simFiles{k_sim}])
    nDataset    = length(AllData);
    
    str = simFiles{k_sim}; % First define the string.
    str1 = str(17:end-4); % Gives a character
    dbl1 = str2double(str1)  % Give a number (double)
    %% analyze
    for k_subjSet = 1:nDataset
        clear glme tbl
        tblALL = AllData{k_subjSet,k_trial};
        V1 = tblALL(:,3);
        V2 = tblALL(:,4);
        OV = V1+V2;
        VD = V1-V2;
        absVD = abs(VD);
        subj= tblALL(:,1);
        RT       = tblALL(:,6);
        logRT    = log(tblALL(:,6));
        Accuracy = logical(tblALL(:,5));
        tbl = table(subj,V1,V2,Accuracy,RT,logRT,VD,OV,absVD);                 
        
        % with full random effect
        glme.CORR        = fitglme(tbl,'Accuracy ~ 1 + OV + absVD + (1+ OV + absVD |subj)','Distribution','Binomial');
        glme.RT        = fitglme(tbl,'RT ~ 1 + OV + absVD+ (1+ OV + absVD |subj)','Distribution','Normal');
        save(['glmeOVeffect/glme_sim' str1 '_subjN' num2str(k_subjSet) '_trial' num2str(k_trial)],'glme','tbl')
    end
end
%% load data
clear all
clc
k_trial = 4;
nSim = 100;
glmelistingt = dir(['glmeOVeffect/*trial' num2str(k_trial) '.mat']);
SuccSimList = length(glmelistingt);
glmeList = {glmelistingt.name};
nDataset    = 7;
count = zeros(1, nDataset+1);
SUMMARY.corrP = nan(nSim,nDataset,3);     SUMMARY.rtP = nan(nSim,nDataset,3);
SUMMARY.corrBeta = nan(nSim,nDataset,3);     SUMMARY.rtBeta = nan(nSim,nDataset,3); 
missingGLME = {};
for k_sim = 1:100
     str = glmeList{k_sim}; % First define the string.
     str1 = str(9:end-18); % Gives a character
     dbl1 = str2double(str1);  % Give a number (double)
    for k_subjSet = 1:nDataset
        currPro = SuccSimList-sum(count)
        resultNAME = ['glme_sim' num2str(k_sim) '_subjN' num2str(k_subjSet) '_trial' num2str(k_trial) '.mat'];
        if isfile(['glmeOVeffect/' resultNAME])
            load(['glmeOVeffect/' resultNAME]);
            count(k_subjSet) = count(k_subjSet) +1;
            SUMMARY.corrP(k_sim,k_subjSet,:) = glme.CORR.Coefficients.pValue;
            SUMMARY.rtP(k_sim,k_subjSet,:) = glme.RT.Coefficients.pValue;
            
            SUMMARY.corrBeta(k_sim,k_subjSet,:) = glme.CORR.Coefficients.Estimate;
            SUMMARY.rtBeta(k_sim,k_subjSet,:) = glme.RT.Coefficients.Estimate;
             
        else
            count(8) = count(8) +1;
            missingGLME{count(8)} = resultNAME;
        end
    end
end
missingGLME = missingGLME';
overviewBeta.OVonCORR= nanmean(squeeze(SUMMARY.corrBeta(:,:,2)));
overviewBeta.OVonRT= nanmean(squeeze(SUMMARY.rtBeta(:,:,2)));

overviewBeta.VDonCORR= nanmean(squeeze(SUMMARY.corrBeta(:,:,3)));
overviewBeta.VDonRT= nanmean(squeeze(SUMMARY.rtBeta(:,:,3)));

overviewP.OVonCORR= nansum(squeeze(SUMMARY.corrP(:,:,2))<0.05);
overviewP.OVonRT= nansum(squeeze(SUMMARY.rtP(:,:,2))<0.05);

overviewP.VDonCORR= nansum(squeeze(SUMMARY.corrP(:,:,3))<0.05);
overviewP.VDonRT= nansum(squeeze(SUMMARY.rtP(:,:,3))<0.05);

estimatedP.OVonCORR= overviewP.OVonCORR./count(1:7);
estimatedP.OVonRT= overviewP.OVonRT./count(1:7);

estimatedP.VDonCORR= overviewP.VDonCORR./count(1:7);
estimatedP.VDonRT= overviewP.VDonRT./count(1:7);

save(['ResultSummary_GLME_20210901_trial_' num2str(k_trial)],'SUMMARY','overviewP','overviewBeta','missingGLME','count','estimatedP')

