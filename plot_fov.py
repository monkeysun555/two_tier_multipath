import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math


VIDEO_LEN = 300
VIDEO_FPS = 30
IS_SAVING = 0
def main():
	mat_contents_1 = sio.loadmat('./traces/output/Video_9_alpha_beta_new.mat')
	mat_contents_2 = sio.loadmat('./traces/output/Video_13_alpha_beta_new.mat')
	trace_data_1 = mat_contents_1['data_alpha_beta'] # array of structures
	trace_data_2 = mat_contents_2['data_alpha_beta'] # array of structures
	# pitch_trace_data = mat_contents['view_angle_pitch_combo'] + 90 # array of structures
	# print(len(yaw_trace_data[0]), len(pitch_trace_data[0]))
	# print(trace_data.T.shape)

	yaw_trace_data_1 = (trace_data_1.T[1]/math.pi)*180 
	pitch_trace_data_1 = (trace_data_1.T[2]/math.pi)*180
	# print(yaw_trace_data.shape, yaw_trace_data[:VIDEO_LEN*VIDEO_FPS])
	yaw_trace_data_1 =  [x for x in yaw_trace_data_1 if not math.isnan(x)]
	pitch_trace_data_1 =  [x for x in pitch_trace_data_1 if not math.isnan(x)]
	# yaw_trace_data.tolist().remove(float('nan'))

	yaw_trace_data_2 = (trace_data_2.T[1]/math.pi)*180
	pitch_trace_data_2 = (trace_data_2.T[2]/math.pi)*180
	# print(yaw_trace_data.shape, yaw_trace_data[:VIDEO_LEN*VIDEO_FPS])
	yaw_trace_data_2 =  [x for x in yaw_trace_data_2 if not math.isnan(x)]
	pitch_trace_data_2 =  [x for x in pitch_trace_data_2 if not math.isnan(x)]

	yaw_trace_data_1 = yaw_trace_data_1[:VIDEO_LEN*VIDEO_FPS]
	yaw_trace_data_2 = yaw_trace_data_2[:VIDEO_LEN*VIDEO_FPS]


	yaw_trace_data_1 = [x + 90 for x in yaw_trace_data_1]
	
	for i in range(len(yaw_trace_data_1)):
		if yaw_trace_data_1[i] >= 180:
			yaw_trace_data_1[i] -= 360
		if yaw_trace_data_1[i] < -180:
			yaw_trace_data_1[i] += 360
	# print(yaw_trace_data_1)



	h = plt.figure(1,figsize=(20,5))

	up_imaginary = [180.0]*VIDEO_LEN
	low_imaginary = [-180.0]*VIDEO_LEN

	pitch_upper = [90.0]*VIDEO_LEN
	pitch_lower = [-90.0]*VIDEO_LEN

	zero_line = [0.0]*VIDEO_LEN
	# plt.plot(display_result[7], display_result[3],'b-', label='Predict Viewport (horizontal central)')
	# plt.plot(display_result[7], [x - 180 for x in display_result[4]],color='cornflowerblue', label='FoV Direction',linewidth=2.5)
	# print(len(np.arange(0,VIDEO_LEN,1.0/float(VIDEO_FPS))))
	# print(len(yaw_trace[:VIDEO_LEN*VIDEO_FPS]))
	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in yaw_trace_data_1[:VIDEO_LEN*VIDEO_FPS]], 'o',\
		color='cornflowerblue', markeredgecolor='cornflowerblue', markersize = 2.8, label='FoV Trace 1')  #,linewidth=2.5)
	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in yaw_trace_data_2[:VIDEO_LEN*VIDEO_FPS]],'o',\
		color='chocolate', markeredgecolor='chocolate', markersize = 2.8, label='FoV Trace 2')  #,linewidth=2.5)
	plt.plot(range(1,VIDEO_LEN+1), up_imaginary,'--', color='gray', linewidth=1.5)
	plt.plot(range(1,VIDEO_LEN+1), low_imaginary,'--', color='gray', linewidth=1.5)
	# plt.plot(range(1,VIDEO_LEN+1), zero_line,'--', color='gray', linewidth=1.5)
	plt.legend(loc='upper right', fontsize = 24, markerscale=3.)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Degree', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, -240 , 450])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(-180, 180+1, 90))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.11, right=0.97)	
	# plt.grid(linestyle='dashed', axis='y', linewidth=1.5,color='gray')
	# h.savefig('fov_trace_1.eps', format='eps', dpi=1000,figsize=(30, 9))



	i = plt.figure(2,figsize=(20,5))
	# plt.plot(display_result[7], display_result[3],'b-', label='Predict Viewport (horizontal central)')
	# plt.plot(display_result[7], [x - 180 for x in display_result[4]],color='cornflowerblue', label='FoV Direction',linewidth=2.5)
	# print(len(np.arange(0,VIDEO_LEN,1.0/float(VIDEO_FPS))))
	# print(len(yaw_trace[:VIDEO_LEN*VIDEO_FPS]))
	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in pitch_trace_data_1[:VIDEO_LEN*VIDEO_FPS]],'o',\
		color='cornflowerblue', markeredgecolor='cornflowerblue', markersize = 2.8, label='FoV Trace 1')  #,linewidth=2.5)
	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in pitch_trace_data_2[:VIDEO_LEN*VIDEO_FPS]],'o',\
		color='chocolate', markeredgecolor='chocolate', markersize = 2.8, label='FoV Trace 2')  #,linewidth=2.5)	
	plt.plot(range(1,VIDEO_LEN+1), pitch_upper,'--', color='gray', linewidth=1.5)
	plt.plot(range(1,VIDEO_LEN+1), pitch_lower,'--', color='gray', linewidth=1.5)
	# plt.plot(range(1,VIDEO_LEN+1), zero_line,'--', color='gray', linewidth=1.5)
	plt.legend(loc='upper right', fontsize = 24, markerscale=3.)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Degree', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, -120 , 270])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(-90, 90+1, 45))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.11, right=0.97)	
	# plt.grid(linestyle='dashed', axis='y', linewidth=1.5,color='gray')
	# i.savefig('fov_trace_2.eps', format='eps', dpi=1000,figsize=(30, 9))

	'''
	j = plt.figure(3,figsize=(20,5))
	# plt.plot(display_result[7], display_result[3],'b-', label='Predict Viewport (horizontal central)')
	# plt.plot(display_result[7], [x - 180 for x in display_result[4]],color='cornflowerblue', label='FoV Direction',linewidth=2.5)
	# print(len(np.arange(0,VIDEO_LEN,1.0/float(VIDEO_FPS))))
	# print(len(yaw_trace[:VIDEO_LEN*VIDEO_FPS]))
	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in yaw_trace_data_2[:VIDEO_LEN*VIDEO_FPS]],'o',\
		color='chocolate', markeredgecolor='chocolate', markersize = 2.8, label='FoV Trace 2')  #,linewidth=2.5)
	plt.plot(range(1,VIDEO_LEN+1), up_imaginary,'--', color='gray', linewidth=1.5)
	plt.plot(range(1,VIDEO_LEN+1), low_imaginary,'--', color='gray', linewidth=1.5)
	# plt.plot(range(1,VIDEO_LEN+1), zero_line,'--', color='gray', linewidth=1.5)
	plt.legend(loc='upper right', fontsize = 30, markerscale=3.)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Degree', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, -240 , 360])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(-180, 180+1, 90))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.09, right=0.97)



	k = plt.figure(4,figsize=(20,5))
	# plt.plot(display_result[7], display_result[3],'b-', label='Predict Viewport (horizontal central)')
	# plt.plot(display_result[7], [x - 180 for x in display_result[4]],color='cornflowerblue', label='FoV Direction',linewidth=2.5)
	# print(len(np.arange(0,VIDEO_LEN,1.0/float(VIDEO_FPS))))
	# print(len(yaw_trace[:VIDEO_LEN*VIDEO_FPS]))
	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in pitch_trace_data_2[:VIDEO_LEN*VIDEO_FPS]],'o',\
		color='chocolate', markeredgecolor='chocolate', markersize = 2.8, label='FoV Trace 2')  #,linewidth=2.5)
	plt.plot(range(1,VIDEO_LEN+1), pitch_upper,'--', color='gray', linewidth=1.5)
	plt.plot(range(1,VIDEO_LEN+1), pitch_lower,'--', color='gray', linewidth=1.5)
	# plt.plot(range(1,VIDEO_LEN+1), zero_line,'--', color='gray', linewidth=1.5)
	plt.legend(loc='upper right', fontsize = 30, markerscale=3.)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Degree', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, -120 , 150])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(-90, 90+1, 45))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.09, right=0.97)
	'''
	h.show()
	i.show()
	# j.show()
	# k.show()
	raw_input()
	if IS_SAVING:
		h.savefig('./figures/fov/yaw.eps', format='eps', dpi=1000, figsize=(30, 10))
		i.savefig('./figures/fov/pitch.eps', format='eps', dpi=1000, figsize=(30, 10))
		# j.savefig('./figures/fov/fov_2_yaw.eps', format='eps', dpi=1000, figsize=(30, 10))
		# k.savefig('./figures/fov/fov_2_pitch.eps', format='eps', dpi=1000, figsize=(30, 10))

if __name__ == '__main__':
	main()
