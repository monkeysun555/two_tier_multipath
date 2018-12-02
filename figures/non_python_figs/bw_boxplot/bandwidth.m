clear;
clear all;


data = [];
group = [];

tempdata = [];
fid = fopen('BW_Trace_5G_0.txt');
tline = fgetl(fid);
while ischar(tline)
    tempdata = [tempdata str2num(tline)];
    tline = fgetl(fid);
end
fclose(fid);

for i=1:2:600
    data = [data (tempdata(i)+tempdata(i+1))/2];
    group = [group 1];
end


tempdata = [];
fid = fopen('BW_Trace_5G_1.txt');
tline = fgetl(fid);
while ischar(tline)
    tempdata = [tempdata str2num(tline)];
    tline = fgetl(fid);
end
fclose(fid);
for i=1:2:600
    data = [data ((tempdata(i)+tempdata(i+1))/2)*1.7 - 520];
    group = [group 2];
end

tempdata = [];
fid = fopen('BW_Trace_5G_2.txt');
tline = fgetl(fid);
while ischar(tline)
    tempdata = [tempdata str2num(tline)];
    tline = fgetl(fid);
end
fclose(fid);
for i=1:2:600
    data = [data ((tempdata(i)+tempdata(i+1))/2)*0.8+80];
    group = [group 3];
end

tempdata = [];
fid = fopen('BW_Trace_5G_3.txt');
tline = fgetl(fid);
while ischar(tline)
    tempdata = [tempdata str2num(tline)];
    tline = fgetl(fid);
end
fclose(fid);
for i=1:2:600
    data = [data (tempdata(i)+tempdata(i+1))/2];
    group = [group 4];
end

tempdata = [];
fid = fopen('BW_Trace_5G_4.txt');
tline = fgetl(fid);
while ischar(tline)
    tempdata = [tempdata str2num(tline)];
    tline = fgetl(fid);
end
fclose(fid);
for i=1:2:600
    data = [data (tempdata(i)+tempdata(i+1))/2];
    group = [group 5];
end


positions = [1 2 3 4 5];
boxplot(data,group, 'positions', positions);
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', 'k','LineWidth',1.2);

% set(gca,'xtick',[mean(positions(1:3)) mean(positions(4:6)) mean(positions(7:9))])
set(gca,'xticklabel',{'BW1','BW2','BW3', 'BW4', 'BW5'},'FontName','Helvetica','FontSize',21)
lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
set(lines, 'Color', 'r');

ylim([0 1000])
set(gca,'YTick',0:200:1000)
ylabel('Throughput(Mbps)')

