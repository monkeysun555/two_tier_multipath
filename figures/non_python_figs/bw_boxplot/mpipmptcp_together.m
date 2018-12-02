clear;
clear all;
% mpip_eth0 = [];
% fid = fopen('mpip_delay_eth0');
% tline = fgetl(fid);
% while ischar(tline)
%     mpip_eth0 = [mpip_eth0 str2num(tline)];
%     tline = fgetl(fid);
% end
% fclose(fid);
% 
% boxplot(mpip_eth0);

data = [];
group = [];


%40_40
fid = fopen('mptcp_40_40');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 1];
    tline = fgetl(fid);
end
fclose(fid);

fid = fopen('mpip_40_40');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 2];
    tline = fgetl(fid);
end
fclose(fid);

fid = fopen('mpipmptcp_40_40');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 3];
    tline = fgetl(fid);
end
fclose(fid);


%delay
fid = fopen('mptcp_delay');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 4];
    tline = fgetl(fid);
end
fclose(fid);

fid = fopen('mpip_delay');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 5];
    tline = fgetl(fid);
end
fclose(fid);

fid = fopen('mpipmptcp_delay');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 6];
    tline = fgetl(fid);
end
fclose(fid);



%  40_20
fid = fopen('mptcp_40_20');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 7];
    tline = fgetl(fid);
end
fclose(fid);

fid = fopen('mpip_40_20_new');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 8];
    tline = fgetl(fid);
end
fclose(fid);

fid = fopen('mpipmptcp_40_20');
tline = fgetl(fid);
while ischar(tline)
    data = [data str2num(tline)];
    group = [group 9];
    tline = fgetl(fid);
end
fclose(fid);



positions = [1 1.25 1.5 2 2.25 2.5  3 3.25 3.5  ];
boxplot(data,group, 'positions', positions);
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', 'k','LineWidth',1.2);

set(gca,'xtick',[mean(positions(1:3)) mean(positions(4:6)) mean(positions(7:9))])
set(gca,'xticklabel',{'Normal','Extra Delay','Bandwidth Limit'},'FontName','Helvetica','FontSize',21)

color = ['r', 'y', 'b', 'r', 'y', 'b','r', 'y', 'b'];
h = findobj(gca,'Tag','Box');
for j=1:length(h)
   pp = patch(get(h(j),'XData'),get(h(j),'YData'),color(j));
   uistack(pp , 'bottom');
end

c = get(gca, 'Children');
hleg1 = legend(c(4:1:2), 'MPTCP/IP', 'TCP/MPIP', 'MPTCP/MPIP','Location','southwest');

legend boxoff;

set(hleg1,'FontName','Helvetica','FontSize',22);
set(gca,'ytick',0:10:80,'yticklabel',0:10:80);



