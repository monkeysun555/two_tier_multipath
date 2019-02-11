% curve plotting
% Generate Q-R relationship
% seq = 'Trolley';
seq = 'Chairlift';
disp('Deriving Q-R Relationship!');
F = @(x,xdata)x(1) + x(2) * log (xdata);
x0 = [1,1]; % random initialization
parameter = zeros(1, 2);
vp_ver = 135;
vp_hor = 135;
c = {[1 0 0], [0 1 0], [0 0 1], [1 1 0], [0 1 1], [.5 .6 .7],[.8 .2 .6]}; % Cell array of colros.
load(strcat([seq, '_VP_using_Tile.mat']));
parameter = zeros(7,2);
bitrate_samples = [];
psnr_samples = [];
locations = [];
figure;
for face = 1:6
    if face == 1
        location = 'Top';
        description = strcat('yaw = 0', char(176), ', pitch = 90', char(176));
        bitrate_sample = BR_top./(vp_ver*vp_ver);
        psnr_sample = PSNR_top;
        plot_location = 1;
    elseif face == 2
        location = 'Bottom';
        description = strcat('yaw = 0', char(176), ', pitch = -90', char(176));
        bitrate_sample = BR_bottom./(vp_ver*vp_ver);
        psnr_sample = PSNR_bottom;
        plot_location = 4;
    elseif face == 3
        location = 'Left';
        description = strcat('yaw = -90', char(176), ', pitch = 0', char(176));
        bitrate_sample = BR_left./(vp_ver*vp_ver);
        psnr_sample = PSNR_left;
        plot_location = 3;
    elseif face == 4
        location = 'Right';
        description = strcat('yaw = 90', char(176), ', pitch = 0', char(176));
        bitrate_sample = BR_right./(vp_ver*vp_ver);
	    psnr_sample = PSNR_right;
        plot_location = 6;
    elseif face == 5
        location = 'Front';
        description = strcat('yaw = 0', char(176), ', pitch = 0', char(176));
        bitrate_sample = BR_front./(vp_ver*vp_ver);
        psnr_sample = PSNR_front;
        plot_location = 2;
    elseif face == 6
        location = 'Back';
        description = strcat('yaw = 180', char(176), ', pitch = 90', char(176));
        bitrate_sample = BR_back./(vp_ver*vp_ver);
        psnr_sample = PSNR_back;
        plot_location = 5;
    else
        disp('Invalid Face Input');
    end
    if face ~= 2 & face ~=1
        bitrate_samples = [bitrate_samples bitrate_sample];
        psnr_samples = [psnr_samples psnr_sample];
    end
    hold on;
    [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0, bitrate_sample, psnr_sample);
    locations = strvcat(locations, [location, ',', ... 
        ' a = ',num2str( round(x(1),3)), ' b = ', num2str(round(x(2),3))]);
    parameter(face, 1:2) = x;
%     subplot(2, 3,plot_location);
    color = cell2mat(c(face));
    scatter(bitrate_sample, psnr_sample, 30, color,'HandleVisibility','off'); plot(bitrate_sample,F(x, bitrate_sample), ':', 'linewidth', 2, 'color', color); 
    if face == 6
        [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0, bitrate_samples, psnr_samples);
        plot(sort(bitrate_samples), F(x, sort(bitrate_samples)), 'linewidth', 2, 'colo','k'); 
        set(gca,'YTick',(36:2:49), 'fontsize',20);
            locations = strvcat(locations, ['Average,', ... 
        ' a = ', num2str(round(x(1),3)), ' b = ', num2str(round(x(2),3))]);
        lgd = legend(locations);
        set(lgd, 'fontsize', 16,'location', 'southeast');
        xlabel('bitrate/degree (kbps/degree)', 'FontSize',20); ylabel('WS-PSNR', 'FontSize', 20); 
        axis([0 max(bitrate_samples)*1.76 36 49]);
        parameter(face+1, 1:2) = x;
        title(strcat([seq, ' sequence average Q-R: ', ... 
        ' a = ', num2str(x(1)), ' b = ', num2str(x(2))]), 'FontSize', 20);
    end
%     title(strcat([location, ': ', description]));
end % face