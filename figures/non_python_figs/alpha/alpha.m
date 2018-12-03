clear;
clear all;

x = [1 2 3 4];
lengthX = length(x);
samplingRateIncrease = 10;
newXSamplePoints = linspace(min(x), max(x), lengthX * samplingRateIncrease);
%%%  Stable %%% Not yet
    %%%  Mean = 690.7
    %%%  Peak = 885
    %%%  std = 211.1
  
    %%%  Buffer_len = [1 2 3 4]
    %%%  Upper_bound = [2 3 4 5]
    %%%  Rate allocation: [158]
    
alpha_21 = [0.933 0.878 0.836 0.800 ];    %% NOT YET 220.302   385.5   514.038  642.55
smoothed_21 = pchip(x, alpha_21, newXSamplePoints);



%%%   Disturbed %%% 
    %%%  mean = 659.67
    %%%  rate_cut = [0.15 0.25 0.5 0.8]
    %%%  Buffer_len = [1 2 3 4]
    %%%  Upper_bound = [2 3 4 5]
    %%%  Rate allocation: [552 721 742 841]

alpha_22 = [0.857 0.776 0.713 0.671];   %% 100% average [Rate0 = 197.899930, Rate1 = 346.324878, Rate2 = 461.766503, Rate3 = 577.208129296]

smoothed_22 = pchip(x, alpha_22, newXSamplePoints);

%%% Unstable  %%%
    %%%  mean 585.31
    %%%  rate_cut = [0.2 0.4 0.6 0.8]
    %%%  Buffer_len = [1 2 3 4]
    %%%  Upper_bound = [2 3 4 5]
    %%%  Rate allocation: [650 734 753 841]
% gamma_23 = [0.824 0.886 0.930 0.955];  %% 100% average [Rate0 = 175.593145, Rate1 = 307.288004, Rate2 = 409.717338, Rate3 = 512.146673294]
% smoothed_23 = spline(x, gamma_23, newXSamplePoints);
MarkerIndices = [1 14 27 40];

imaginary = [1 1 1 1];
figure(1);
% hold on;
h = plot(newXSamplePoints, smoothed_21, '-', newXSamplePoints, smoothed_22,'-', 'LineWidth',2);
hold on;          
% plot(1:4, imaginary, '--','Color', [0.5 0.5 0.5], 'LineWidth',1.5);
c = get(h,'Color');
mymarkers1 = plot(newXSamplePoints(MarkerIndices), smoothed_21(MarkerIndices), 'o','Color',c{1},'MarkerSize',8,'MarkerFaceColor',c{1});    
mymarkers2 = plot(newXSamplePoints(MarkerIndices), smoothed_22(MarkerIndices), 'o','Color',c{2},'MarkerSize',8,'MarkerFaceColor',c{2}); 
% mymarkers3 = plot(newXSamplePoints(MarkerIndices), smoothed_23(MarkerIndices), 'o','Color',c{3},'MarkerSize',8,'MarkerFaceColor',c{3});    
% plot(0:len-1, Total, 'Color', [0.5 0.5 0.5 ],'LineWidth',1.2);
h_legend = legend('FoV Trace 1', 'FoV Trace 2','Location','northeast');
set(h_legend, 'FontSize', 22);
% title('5G Trace Gamma Curve','FontName','Times New Roman','FontWeight','Bold','FontSize',20)
xlabel('Prediction Horizon (s)','FontName','Helvetica','FontSize',22);
ylabel('{\boldmath$\alpha$}','Interpreter','latex','FontName','Helvetica','FontSize',26,'FontWeight','bold');
set(gca,'xtick',0:1:4,'xticklabel',0:1:4, 'FontSize',22);
set(gca,'ytick',0.6:0.1:1,'yticklabel',0.6:0.1:1,'FontSize',22);
axis([1 4 0.6 1])% axis auto equal

xt1 = [1 1.9 2.9 3.75];
xt2 = [1 2.0 2.9 3.6];
% xt3 = [1 1.8 2.8 3.6];

yt1 = alpha_21+0.015;
yt2 = alpha_22+0.015;
% yt3 = gamma_23+0.01;
yt1(4) = yt1(4)+0.01;
yt2(4) = yt2(4)+0.01;
str1 = {};
str2 = {};
% str3 = {};
for i = 1:length(alpha_21)
    str1{i} = num2str(alpha_21(i));
    str2{i} = num2str(alpha_22(i));
%     str3{i} = num2str(gamma_23(i));
end

text(xt1,yt1,str1,'FontSize',22);
text(xt2,yt2,str2,'FontSize',22);
% text(xt3,yt3,str3,'FontSize',22);

