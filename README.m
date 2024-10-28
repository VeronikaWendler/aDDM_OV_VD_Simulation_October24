%% aDDM_simulation.m
% Main function used to simulate data for different settings.
% Simulation of aDDM to account for magnitude effect
% - In this simple simulation, five parameters are defined based on previous literatures
% 1. Drift rate
% 2. Va and Vb
% 3. SD of noise
% 4. threshold
% 5. Probability of looking at option A
% 6. theta: Impact of attention 
% - Simulate data for 10-70 subjects with 20-50 trials/conditions. 
% 1. Va = 3, Vb = 1 (OV = 4, VD =2)
% 2. Va = 6, Vb = 2 (OV = 8, VD =4)
% 3. Va = 8, Vb = 4 (OV = 12, VD =4)
% 4. Va = 9, Vb = 7 (OV = 16, VD =2)

% - the results are stored in "simulation_20210824"

%% EvidenceAccumulate_v2.m
% - Main script used to perform evidence accumulation in each trial

%% GLMEanlaysis_simuData.m
% analyze data with GLME
% - the results are stored in folder "glmeOVeffect" and in the current folder 
% Note: It is not necessary to save GLME result for each simulation to generate the summary of GLME results (and save them in ResultSummary_GLME_20210901_trial_X.mat). To do so, you could integrate two sections in the script.  

%% SummaryGLME.m
% plot power for each setting (X subjects and N trials/condition).
