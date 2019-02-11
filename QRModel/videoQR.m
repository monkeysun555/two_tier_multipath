clear;
draw_side_by_side = 1;
bt_ver = 180;
bt_hor = 360;
F = @(x,xdata)x(1) + x(2) * log (xdata);
x0 = [1,1]; % random initialization
parameter = zeros(1, 2);
figure;
if draw_side_by_side
    % Chairlift
    seq = 'Trolley';
    load('Trolley_video_QR.mat');
    [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0, bitrate_sample./(bt_ver*bt_hor), psnr_sample);
    subplot(1,2,1);
    scatter(bitrate_sample./(bt_ver*bt_hor), psnr_sample); hold on; 
    plot(bitrate_sample./(bt_ver*bt_hor), psnr_curve);
    set(gca,'YLim',[38 47], 'YTick',(38:2:46), 'FontSize', 20)
    xlabel('bitrate/degree (kbps/degree)', 'fontsize', 20); ylabel('WS-PSNR', 'fontsize', 20); 
    title(strcat([seq, ' BT Q-R Fitting: ', ... 
        ' a = ', num2str(x(1)), ' b = ', num2str(x(2))]), 'fontsize', 20);
    % Chairlift
    seq = 'Chairlift';
    load('Chairlift_video_QR.mat');
    [x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0, bitrate_sample./(bt_ver*bt_hor), psnr_sample);
    subplot(1,2,2);
    scatter(bitrate_sample./(bt_ver*bt_hor), psnr_sample); hold on; 
    plot(bitrate_sample./(bt_ver*bt_hor), psnr_curve);
    set(gca,'YLim',[38 47], 'YTick',(38:2:47), 'FontSize', 20)
    xlabel('bitrate/degree (kbps/degree)', 'fontsize', 20); ylabel('WS-PSNR', 'fontsize', 20); 
    title(strcat([seq, ' BT Q-R Fitting: ', ... 
        ' a = ', num2str(x(1)), ' b = ', num2str(x(2))]), 'fontsize', 20); 

end