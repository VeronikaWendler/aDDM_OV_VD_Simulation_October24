% - Main script used to perform evidence accumulation in each trial
function [Choice, RT, E, gazeA, gazeB, nSwitch]= EvidenceAccumulate_v2(Va,Vb,params,threshold)

%% pre-determine fixations
Nfix = 70; % assume 70 fixations before decision

% list of fixation. Fixate at A (better option): FixA =1 
if rand(Nfix,1)>0.5
    FixA = repmat([0 1],1,Nfix/2);
else
    FixA = repmat([1 0],1,Nfix/2);
end

FixDur = round(lognrnd(6, 0.6,Nfix,1)); %fixation period Krajbich's 2010 and Gluth's 2020

dvPeriod = cell(1,Nfix);
k_fix =0;
dv  = params(3) * ((Va-params(1)*Vb) * (FixA==1) + (params(1)*Va-Vb) * (FixA==0)); % general evidence for each starting point of a fixation (e.g., the first time point of fixating at optionA)

while k_fix< Nfix 
    k_fix = k_fix+1;
    err = normrnd(0,params(2),1,FixDur(k_fix));
    dvPeriod{k_fix,1} = dv(k_fix)+err; 
    Fix.A{k_fix,1} = repmat(FixA(k_fix),1,FixDur(k_fix));
    if k_fix ==1
        Fix.duration{k_fix,1} = 1:FixDur(k_fix);
    else
        Fix.duration{k_fix,1} = (sum(FixDur(1:k_fix-1))+1):sum(FixDur(1:k_fix));
    end
end
dvALL = [dvPeriod{1:k_fix}];
sumdvALL = cumsum([dvPeriod{1:k_fix}],2);
ID = find(abs(sumdvALL)>=threshold(1)); % find the dv that is larger than threshold (assuming threshold is symmetric)
FixAALL  = [Fix.A{1:k_fix}];
FixDurALL= [Fix.duration{1:k_fix}];
FixationInfo = [FixAALL(1:ID(1));FixDurALL(1:ID(1))];
E            = sumdvALL(1:ID(1));
gazeA        = sum(FixationInfo(1,:) == 1);
gazeB        = sum(FixationInfo(1,:) == 0);
nSwitch      = 0;
nSwitch      = sum(diff(FixationInfo(1,:))~=0);

Choice = E(end)>1; % 1: correct; 0: incorrect
RT = length(E)+params(4);



