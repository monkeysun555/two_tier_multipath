% Generate Q-R relationship
bitrate = [17245.11 7783.90 4302.52 2498.74 ];
wspsnr = [47.70 45.56 43.36 40.76  ];
disp('Deriving Q-R Relationship!');
F = @(x,xdata)x(1) + x(2) * log (xdata);
x0 = [1,1]; % random initialization
parameter = zeros(1, 2);
psnr_sample = wspsnr;
bitrate_sample = bitrate;  
[x,resnorm,~,exitflag,output] = lsqcurvefit(F,x0, bitrate_sample, psnr_sample);
figure; scatter(bitrate_sample, psnr_sample); hold on; plot(bitrate_sample,F(x, bitrate_sample));
xlabel('bitrate (kbps)'); ylabel('WS-PSNR'); title(strcat(['Full-360 Video: ', ' a = ', num2str(x(1)), ' b = ', num2str(x(2))]));
parameter = x;
