clear;
clear all;

x = [1 2 3 4];
lengthX = length(x);
samplingRateIncrease = 10;
newXSamplePoints = linspace(min(x), max(x), lengthX * samplingRateIncrease);


gamma_21 = [0.956 1.0 1.0 1.0];  %% Sun noon, REFINE
smoothed_21 = pchip(x, gamma_21, newXSamplePoints);


gamma_22 = [0.883 0.987 1.0 1.0];  %% Sun noon,REFINE
smoothed_22 = pchip(x, gamma_22, newXSamplePoints);

gamma_23 = [0.823 0.942 0.978 1.0];  %% Sun noon, REFINE
smoothed_23 = pchip(x, gamma_23, newXSamplePoints);

gamma_24 = [0.780 0.913 0.970 1.0];  %% Sun noon, REFINE
smoothed_24 = pchip(x, gamma_24, newXSamplePoints);

gamma_25 = [0.669 0.797 0.862 0.893];  %% Sun noon, REFINE
smoothed_25 = pchip(x, gamma_25, newXSamplePoints);

MarkerIndices = [1 14 27 40];


imaginary = [1 1 1 1];
figure(1);
% hold on;
h = plot(newXSamplePoints, smoothed_21, '-', newXSamplePoints, smoothed_22,'-', newXSamplePoints, smoothed_23,'-', newXSamplePoints, smoothed_24,'-', newXSamplePoints, smoothed_25,'-', 'LineWidth',2);
hold on;          
plot(1:4, imaginary, '--','Color', [0.5 0.5 0.5], 'LineWidth',1.5);
c = get(h,'Color');
mymarkers1 = plot(newXSamplePoints(MarkerIndices), smoothed_21(MarkerIndices), 'o','Color',c{1},'MarkerSize',8,'MarkerFaceColor',c{1});    
mymarkers2 = plot(newXSamplePoints(MarkerIndices), smoothed_22(MarkerIndices), 'o','Color',c{2},'MarkerSize',8,'MarkerFaceColor',c{2}); 
mymarkers3 = plot(newXSamplePoints(MarkerIndices), smoothed_23(MarkerIndices), 'o','Color',c{3},'MarkerSize',8,'MarkerFaceColor',c{3});    
mymarkers4 = plot(newXSamplePoints(MarkerIndices), smoothed_24(MarkerIndices), 'o','Color',c{4},'MarkerSize',8,'MarkerFaceColor',c{4});    
mymarkers5 = plot(newXSamplePoints(MarkerIndices), smoothed_25(MarkerIndices), 'o','Color',c{5},'MarkerSize',8,'MarkerFaceColor',c{5});    

% plot(0:len-1, Total, 'Color', [0.5 0.5 0.5 ],'LineWidth',1.2);
h_legend = legend({'BW1($$c_{v1}\approx$$0.030)', 'BW2($$c_{v2}\approx$$0.119)', 'BW3($$c_{v3}\approx$$0.215)' ,'BW4($$c_{v4}\approx$$0.313)','BW5($$c_{v5}\approx$$0.473)'},'Interpreter','latex','Location','southeast');
set(h_legend, 'FontSize', 22);
% title('5G Trace Gamma Curve','FontName','Times New Roman','FontWeight','Bold','FontSize',20)
xlabel('ET Prefetching Buffer Length  (s)','FontName','Helvetica','FontSize',22);
ylabel('{\boldmath$\gamma$}','Interpreter','latex','FontName','Helvetica','FontSize',26,'FontWeight','bold');
set(gca,'xtick',0:1:4,'xticklabel',0:1:4, 'FontSize',22);
set(gca,'ytick',0.5:0.1:1.05,'yticklabel',0.5:0.1:1.05,'FontSize',22);
axis([1 4 0.5 1.05])% axis auto equal

xt1 = [1.02 1.95 2.95 3.9];
xt2 = [1.02 2.07];
xt3 = [1.02 1.6 2.58];
xt4 = [1.06 2.05 3.03];
xt5 = [1.06 2.05 3.03 3.6];

yt1 = gamma_21+0.015;
yt2 = gamma_22+0.015;
yt3 = gamma_23+0.015;
yt4 = gamma_24-0.01;
yt5 = gamma_25-0.01;

str1 = {};
str2 = {};
str3 = {};
for i = 1:length(gamma_21)
    str1{i} = num2str(gamma_21(i));
    str2{i} = num2str(gamma_22(i));
    str3{i} = num2str(gamma_23(i));
    str4{i} = num2str(gamma_24(i));
    str5{i} = num2str(gamma_25(i));

end
% str1 = {'0.998','1.0','1.0','1.0'};
yt1(1) = yt1(1);
yt2(2) = yt2(2)-0.02;
yt3(2) = yt3(2)-0.01;
yt3(3) = yt3(3)-0.016;
yt4(2) = yt4(2)-0.005;
yt4(3) = yt4(3)-0.01;
yt5(2) = yt5(2)-0.005;
yt5(3) = yt5(3)-0.01;
yt5(4) = yt5(4)+0.02;
text(xt1,yt1,str1,'FontSize',22);
text(xt2,yt2(1:2),str2(1:2),'FontSize',22);
text(xt3,yt3(1:3),str3(1:3),'FontSize',22);
text(xt4,yt4(1:3),str4(1:3),'FontSize',22);
text(xt5,yt5,str5,'FontSize',22);

