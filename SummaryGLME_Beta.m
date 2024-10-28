colorset = [1 0 0; ...
    0.8 0.2 0; ...
    0.6 0.1 0; ...
    0.4 0.05 0];

for k_trialN = 1:4
load(['ResultSummary_GLME_20210901_trial_' num2str(k_trialN) '.mat']);
% plot estimated p-value of OV effect on each variable of interest
figure(1); hold on
subplot(1,3,1); hold on
plot(1:7,overviewBeta.OVonCORR,'-o', ...
    'MarkerFaceColor',colorset(k_trialN,:), ...
    'MarkerEdgeColor',colorset(k_trialN,:), ...
    'Color', colorset(k_trialN,:), ...
    'LineWidth',3, ...
    'MarkerSize',10); hold on
xticks(1:7);
xticklabels({'10','20','30','40','50','60','70'});
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',18)
legend({'trial n = 20' 'trial n = 30' 'trial n = 40' 'trial n = 50'},'FontSize',14)
ylim([-0.07 -0.03])
xlabel('Sample size', 'FontSize', 16)
ylabel('Estimated Beta', 'FontSize', 16)
title('Accuracy', 'FontSize', 16)

subplot(1,3,2); hold on
plot(1:7,overviewBeta.OVonRT,'-o', ...
    'MarkerFaceColor',colorset(k_trialN,:), ...
    'MarkerEdgeColor',colorset(k_trialN,:), ...
    'Color', colorset(k_trialN,:), ...
    'LineWidth',3, ...
    'MarkerSize',10); hold on
xticks(1:7);
xticklabels({'10','20','30','40','50','60','70'});
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',18)
ylim([-50 -40])
xlabel('Sample size', 'FontSize', 16)
ylabel('Estimated Beta', 'FontSize', 16)
title('RT', 'FontSize', 16)


% subplot(1,3,3); hold on
% plot(1:7,overviewBeta.OVonCORRadv,'-o', ...
%     'MarkerFaceColor',colorset(k_trialN,:), ...
%     'MarkerEdgeColor',colorset(k_trialN,:), ...
%     'Color', colorset(k_trialN,:), ...
%     'LineWidth',3, ...
%     'MarkerSize',10); hold on
% xticks(1:7);
% xticklabels({'10','20','30','40','50','60','70'});
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',18)
% ylim([-0.12 -0.05])
% xlabel('Sample size', 'FontSize', 16)
% ylabel('Estimated Beta', 'FontSize', 16)
% title('Accuracy (control Dwell%)', 'FontSize', 16)


% plot estimated p-value of VD effect on each variable of interest
figure(2); hold on
subplot(1,3,1); hold on
plot(1:7,overviewBeta.VDonCORR,'-o', ...
    'MarkerFaceColor',colorset(k_trialN,:), ...
    'MarkerEdgeColor',colorset(k_trialN,:), ...
    'Color', colorset(k_trialN,:), ...
    'LineWidth',3, ...
    'MarkerSize',10); hold on
xticks(1:7);
xticklabels({'10','20','30','40','50','60','70'});
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',18)
ylim([0.3 0.5])
xlabel('Sample size', 'FontSize', 16)
ylabel('Estimated Beta', 'FontSize', 16)
legend({'trial n = 20' 'trial n = 30' 'trial n = 40' 'trial n = 50'},'FontSize',14)
title('Accuracy', 'FontSize', 16)

subplot(1,3,2); hold on
plot(1:7,overviewBeta.VDonRT,'-o', ...
    'MarkerFaceColor',colorset(k_trialN,:), ...
    'MarkerEdgeColor',colorset(k_trialN,:), ...
    'Color', colorset(k_trialN,:), ...
    'LineWidth',3, ...
    'MarkerSize',10); hold on
xticks(1:7);
xticklabels({'10','20','30','40','50','60','70'});
a = get(gca,'XTickLabel');
set(gca,'XTickLabel',a,'fontsize',18)
ylim([-70 -20])
xlabel('Sample size', 'FontSize', 16)
ylabel('Estimated Beta', 'FontSize', 16)
title('RT', 'FontSize', 16)


% subplot(1,3,3); hold on
% plot(1:7,overviewBeta.VDonCORRadv,'-o', ...
%     'MarkerFaceColor',colorset(k_trialN,:), ...
%     'MarkerEdgeColor',colorset(k_trialN,:), ...
%     'Color', colorset(k_trialN,:), ...
%     'LineWidth',3, ...
%     'MarkerSize',10); hold on
% xticks(1:7);
% xticklabels({'10','20','30','40','50','60','70'});
% a = get(gca,'XTickLabel');
% set(gca,'XTickLabel',a,'fontsize',18)
% ylim([0.4 0.7])
% xlabel('Sample size', 'FontSize', 16)
% ylabel('Estimated Beta', 'FontSize', 16)
% title('Accuracy (control Dwell%)', 'FontSize', 16)

end

figure(1);
sgtitle('OV effect on ...') 
figure(2);
sgtitle('VD effect on ...') 
