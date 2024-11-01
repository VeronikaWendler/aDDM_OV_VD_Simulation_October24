colorset = [1 0 0; ...
    0.8 0.2 0; ...
    0.6 0.1 0; ...
    0.4 0.05 0; ...
    0 0 1];

for k_trialN = 2:5
    % Load GLME and Ensemble results
    load(['Derivatives/Results_summary_GlmeEns/ResultSummary_GLME_' num2str(k_trialN) '.mat']);
    load(['Derivatives/Results_summary_GlmeEns/ResultSummary_Ensemble_' num2str(k_trialN) '.mat']);

    % Plot estimated p-value of OV effect on each variable of interest (GLME)
    figure(1); hold on;
    
    % OV on CORR
    subplot(1,8,1); hold on;
    plot(1:6,estimatedP.OVonCORR,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    legend({'trial n = 20' 'trial n = 30' 'trial n = 40' 'trial n = 50' 'trial n = 60'},'FontSize',14);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('Accuracy', 'FontSize', 16);

    % OV on RT
    subplot(1,8,2); hold on;
    plot(1:6,estimatedP.OVonRT,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('RT', 'FontSize', 16);

    % OV on Dwell
    subplot(1,8,3); hold on;
    plot(1:6,estimatedP.OVonDwell,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('Dwell', 'FontSize', 16);

    % OV on CORR_adv
    subplot(1,8,4); hold on;
    plot(1:6,estimatedP.OVonCORRadv,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('CORRadv', 'FontSize', 16);

    % OV on Dwell_rt
    subplot(1,8,5); hold on;
    plot(1:6,estimatedP.OVonDwell_rt,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('Dwell_rt', 'FontSize', 16);

    % OV on RT_dwell
    subplot(1,8,6); hold on;
    plot(1:6,estimatedP.OVonRT_dwell,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('RT_dwell', 'FontSize', 16);

    % OV on CORR_dwellbo
    subplot(1,8,7); hold on;
    plot(1:6,estimatedP.OVonCORR_dwellbo,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('CORR_dwellbo', 'FontSize', 16);

    % OV on Non_Lin_CORR
    subplot(1,8,8); hold on;
    plot(1:6,estimatedP.OVonCORRNon_Lin_CORR,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.3:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'30','40','50','60','70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.3 1.02]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('Non-Lin CORR', 'FontSize', 16);

    % Plot estimated p-value of VD effect on each variable of interest (GLME)
    figure(2); hold on;

    % VD on CORR
    subplot(1,8,1); hold on;
    plot(1:6,estimatedP.VDonCORR,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    legend({'trial n = 20' 'trial n = 30' 'trial n = 40' 'trial n = 50' 'trial n = 60'},'FontSize',14);
    title('Accuracy', 'FontSize', 16);

    % VD on RT
    subplot(1,8,2); hold on;
    plot(1:6,estimatedP.VDonRT,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('RT', 'FontSize', 16);

    % VD on Dwell
    subplot(1,8,3); hold on;
    plot(1:6,estimatedP.VDonDwell,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('Dwell', 'FontSize', 16);

    % VD on CORR_adv
    subplot(1,8,4); hold on;
    plot(1:6,estimatedP.VDonCORRadv,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('CORRadv', 'FontSize', 16);

    % VD on Dwell_rt
    subplot(1,8,5); hold on;
    plot(1:6,estimatedP.VDonDwell_rt,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('Dwell_rt', 'FontSize', 16);

    % VD on RT_dwell
    subplot(1,8,6); hold on;
    plot(1:6,estimatedP.VDonRT_dwell,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('RT_dwell', 'FontSize', 16);

    % VD on CORR_dwellbo
    subplot(1,8,7); hold on;
    plot(1:6,estimatedP.VDonCORR_dwellbo,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('CORR_dwellbo', 'FontSize', 16);

    % VD on Non_Lin_CORR
    subplot(1,8,8); hold on; 
    plot(1:6,estimatedP.VDonCORRNon_Lin_CORR,'-o', ...
        'MarkerFaceColor',colorset(k_trialN,:), ...
        'MarkerEdgeColor',colorset(k_trialN,:), ...
        'Color', colorset(k_trialN,:), ...
        'LineWidth',3, ...
        'MarkerSize',10); hold on;
    xticks(1:6); yticks(0.7:0.1:1.1); 
    xticklabels({'10','20','30','40','50','60','70'});
    yticklabels({'70','80','90','100'});
    a = get(gca,'XTickLabel');
    set(gca,'XTickLabel',a,'fontsize',18);
    ylim([0.7 1.1]);
    xlabel('Sample size', 'FontSize', 16);
    ylabel('Power', 'FontSize', 16);
    title('Non-Lin CORR', 'FontSize', 16);

    
end

figure(1);
sgtitle('OV effect on GLME models') 
figure(2);
sgtitle('VD effect on GLME models') 






% colorset = [1 0 0; ...
%     0.8 0.2 0; ...
%     0.6 0.1 0; ...
%     0.4 0.05 0; ...
%     0 0 1];
% 
% % OV effects
% figure; hold on;
% for k_trialN = 2:5
%     load(['Derivatives/Results_summary_2/ResultSummary_GLME_20210901_trial_' num2str(k_trialN) '.mat']);
%     
%     % OV on CORR (Figure 1)
%     figure(1); hold on;
%     plot(1:6, estimatedP.OVonCORR, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.3:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.02]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('OV effect on Accuracy', 'FontSize', 16);
%     
%     % OV on RT (Figure 2)
%     figure(2); hold on;
%     plot(1:6, estimatedP.OVonRT, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.3:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.02]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('OV effect on RT', 'FontSize', 16);
%     
%     % OV on Dwell (Figure 3)
%     figure(3); hold on;
%     plot(1:6, estimatedP.OVonDwell, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.3:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.02]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('OV effect on Dwell', 'FontSize', 16);
% 
%     % OV on CORR_adv (Figure 4)
%     figure(4); hold on;
%     plot(1:6, estimatedP.OVonCORRadv, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.3:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.02]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('OV effect on CORR_adv', 'FontSize', 16);
%     
%     % OV on Dwell_rt (Figure 5)
%     figure(5); hold on;
%     plot(1:6, estimatedP.OVonDwell_rt, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.3:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.02]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('OV effect on Dwell_rt', 'FontSize', 16);
%     
%     % OV on RT_dwell (Figure 6)
%     figure(6); hold on;
%     plot(1:6, estimatedP.OVonRT_dwell, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.3:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.02]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('OV effect on RT_dwell', 'FontSize', 16);
% 
%     % OV on CORR_dwellbo (Figure 7)
%     figure(7); hold on;
%     plot(1:6, estimatedP.OVonCORR_dwellbo, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.3:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.02]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('OV effect on CORR_dwellbo', 'FontSize', 16);
% 
% end
% 
% % VD effects
% figure; hold on;
% for k_trialN = 2:5
%     load(['Derivatives/Results_summary_2/ResultSummary_GLME_20210901_trial_' num2str(k_trialN) '.mat']);
%     
%     % VD on CORR (Figure 8)
%     figure(8); hold on;
%     plot(1:6, estimatedP.VDonCORR, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.7:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.1]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('VD effect on Accuracy', 'FontSize', 16);
%     
%     % VD on RT (Figure 9)
%     figure(9); hold on;
%     plot(1:6, estimatedP.VDonRT, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.7:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.1]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('VD effect on RT', 'FontSize', 16);
%     
%     % VD on Dwell (Figure 10)
%     figure(10); hold on;
%     plot(1:6, estimatedP.VDonDwell, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.7:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.1]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('VD effect on Dwell', 'FontSize', 16);
% 
%     % VD on CORR_adv (Figure 11)
%     figure(11); hold on;
%     plot(1:6, estimatedP.VDonCORRadv, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.7:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.1]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('VD effect on CORR_adv', 'FontSize', 16);
%     
%     % VD on Dwell_rt (Figure 12)
%     figure(12); hold on;
%     plot(1:6, estimatedP.VDonDwell_rt, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.7:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.1]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('VD effect on Dwell_rt', 'FontSize', 16);
% 
%     % VD on RT_dwell (Figure 13)
%     figure(13); hold on;
%     plot(1:6, estimatedP.VDonRT_dwell, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.7:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.1]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('VD effect on RT_dwell', 'FontSize', 16);
% 
%     % VD on CORR_dwellbo (Figure 14)
%     figure(14); hold on;
%     plot(1:6, estimatedP.VDonCORR_dwellbo, '-o', ...
%         'MarkerFaceColor', colorset(k_trialN,:), ...
%         'MarkerEdgeColor', colorset(k_trialN,:), ...
%         'Color', colorset(k_trialN,:), ...
%         'LineWidth', 3, ...
%         'MarkerSize', 10);
%     xticks(1:6); yticks(0.7:0.1:1.1); 
%     xticklabels({'10', '20', '30', '40', '50', '60'});
%     ylim([0.3 1.1]);
%     xlabel('Sample size', 'FontSize', 16);
%     ylabel('Power', 'FontSize', 16);
%     title('VD effect on CORR_dwellbo', 'FontSize', 16);
% end