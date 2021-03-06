import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import math
import utilities as uti
import pickle 


VIDEO_LEN = 450
VIDEO_FPS = 30
IS_SAVING = 1
ALPHA_AHEAD = 0.5

REVISION = 1
TSINGHUA_TRACE = 1

USER_1 = 0
USER_2 = 6
# For plot yaw, shit by 180; otherwise, the point is changing around -180 and 180. 
SHIFT = 1	# For ploting	
DO_STATISTIC = 0	# For prob counting
def main():

	if not REVISION:
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
	else:
		if not TSINGHUA_TRACE:
			contents_1 = pickle.load(open("./traces/output/gt_theta_phi_user1.p", "rb"))
			contents_2 = pickle.load(open("./traces/output/gt_theta_phi_user7.p", "rb"))

			yaw_trace_data_1 = (contents_1['gt_theta']/math.pi)*180.0
			pitch_trace_data_1 = (contents_1['gt_phi']/math.pi)*180.0 - 90.0
			yaw_trace_data_1 =  [x for x in yaw_trace_data_1 if not math.isnan(x)]
			pitch_trace_data_1 =  [x for x in pitch_trace_data_1 if not math.isnan(x)]

			yaw_trace_data_2 = (contents_2['gt_theta']/math.pi)*180.0
			pitch_trace_data_2 = (contents_2['gt_phi']/math.pi)*180.0 - 90.0
			yaw_trace_data_2 =  [x for x in yaw_trace_data_2 if not math.isnan(x)]
			pitch_trace_data_2 =  [x for x in pitch_trace_data_2 if not math.isnan(x)]

		else:
			contents_1 = pickle.load(open("./traces/output/gt_theta_phi_vid_3.p", "rb"))[1]
			contents_2 = pickle.load(open("./traces/output/gt_theta_phi_vid_3.p", "rb"))[2]
			contents_3 = pickle.load(open("./traces/output/gt_theta_phi_vid_3.p", "rb"))[3]

			yaw_vid_1 = (contents_1['gt_theta']/math.pi)*180.0
			pitch_vid_1 = (contents_1['gt_phi']/math.pi)*180.0 - 90.0

			yaw_vid_2 = (contents_2['gt_theta']/math.pi)*180.0
			pitch_vid_2 = (contents_2['gt_phi']/math.pi)*180.0 - 90.0

			yaw_vid_3 = (contents_3['gt_theta']/math.pi)*180.0
			pitch_vid_3 = (contents_3['gt_phi']/math.pi)*180.0 - 90.0

			# print(len(yaw_vid_1[USER_1]), len(yaw_vid_2[USER_1]))
			
			# print(yaw_vid_1[USER_1], len(yaw_vid_1[USER_1]))
			yaw_trace_data_1 = yaw_vid_1[USER_1].tolist() + yaw_vid_2[USER_1].tolist() + yaw_vid_3[USER_1].tolist()
			pitch_trace_data_1 = pitch_vid_1[USER_1].tolist() + pitch_vid_2[USER_1].tolist() + pitch_vid_3[USER_1].tolist()

			yaw_trace_data_2 = yaw_vid_1[USER_2].tolist() + yaw_vid_2[USER_2].tolist() + yaw_vid_3[USER_2].tolist()
			pitch_trace_data_2 = pitch_vid_1[USER_2].tolist() + pitch_vid_2[USER_2].tolist() + pitch_vid_3[USER_2].tolist()


			# For statistic among 6 ET directions
			# 0: top, 1: bottom, 2: front, 3: back, 4:left, 5:right
			# pitch: -90 to 90
			# yaw: -180 to 180
			if DO_STATISTIC:
				total_prob = [];
				for u_id in range(len(yaw_vid_1)):
					temp_yaw_trace = yaw_vid_1[u_id].tolist() + yaw_vid_2[u_id].tolist() + yaw_vid_3[u_id].tolist()
					temp_pitch_trace = pitch_vid_1[u_id].tolist() + pitch_vid_2[u_id].tolist() + pitch_vid_3[u_id].tolist()
					# print(min(temp_yaw_trace), max(temp_yaw_trace))
					temp_et_count = [0.0]*6
					temp_prob = []

					for i in range(len(temp_yaw_trace)):
						if temp_pitch_trace[i] > 45:
							temp_et_count[0] += 1
						elif temp_pitch_trace[i] < -45:
							temp_et_count[1] += 1
						else:
							temp_yaw_trace[i]  += 180.0
							if temp_yaw_trace[i] >= 45.0 and temp_yaw_trace[i] < 135.0:
								temp_et_count[5] += 1
							elif temp_yaw_trace[i] >= 135.0 and temp_yaw_trace[i] < 225.0:
								temp_et_count[3] += 1
							elif temp_yaw_trace[i] >= 225.0 and temp_yaw_trace[i] < 315.0:
								temp_et_count[4] += 1
							else:
								temp_et_count[2] += 1
					# print(temp_et_count)
					temp_prob = [count/sum(temp_et_count) for count in temp_et_count]
					total_prob.append(temp_prob)
				over_prob = [0.0]*6;
				for j in range(6):
					over_prob[j] = sum([prob[j]/len(total_prob) for prob in total_prob])
				print(over_prob)
				assert round(sum(over_prob),3) == 1.0
			# n, bins, patches=plt.hist(yaw_trace_data_1)
			# plt.show()
			# raw_input()
			# print(yaw_trace_data_1, len(yaw_trace_data_1))
			# print(pitch_trace_data_1, len(pitch_trace_data_1))

	yaw_trace_data_1 = yaw_trace_data_1[:VIDEO_LEN*VIDEO_FPS]
	yaw_trace_data_2 = yaw_trace_data_2[:VIDEO_LEN*VIDEO_FPS]


	for i in range(len(yaw_trace_data_1)):
		if SHIFT: 
			yaw_trace_data_1[i]  += 180.0
		if yaw_trace_data_1[i] >= 180:
			yaw_trace_data_1[i] -= 360
		if yaw_trace_data_1[i] < -180:
			yaw_trace_data_1[i] += 360

	for i in range(len(yaw_trace_data_2)):
		if SHIFT:
			yaw_trace_data_2[i] += 180.0
		if yaw_trace_data_2[i] >= 180:
			yaw_trace_data_2[i] -= 360
		if yaw_trace_data_2[i] < -180:
			yaw_trace_data_2[i] += 360


	yaw_trace = yaw_trace_data_1
	pitch_trace = pitch_trace_data_1
	accuracy_list1 = [0.0]*4
	for ahead_time in range(4):
		sum_accuracy = 0.0
		count = 0
		for i in range(2, VIDEO_LEN - ahead_time, 1):
			yaw_predict_value, yaw_predict_quan = uti.predict_yaw_trun(yaw_trace, i - ALPHA_AHEAD, i+ahead_time)
			pitch_predict_value, pitch_predict_quan = uti.predict_pitch_trun(pitch_trace, i - ALPHA_AHEAD, i+ahead_time)

			start_frame_index = (i + ahead_time)*VIDEO_FPS
			end_frame_index = (i + ahead_time + 1)*VIDEO_FPS
			real_yaw_trace = yaw_trace[start_frame_index:end_frame_index]
			real_pitch_trace = pitch_trace[start_frame_index:end_frame_index]


			el_accuracy = uti.cal_accuracy(yaw_predict_value, yaw_predict_quan, pitch_predict_value, pitch_predict_quan, real_yaw_trace, real_pitch_trace, 1.0)
			sum_accuracy += el_accuracy
			count += 1
		accuracy_list1[ahead_time] = sum_accuracy/count

	yaw_trace = yaw_trace_data_2
	pitch_trace = pitch_trace_data_2
	accuracy_list2 = [0.0]*4
	for ahead_time in range(4):
		sum_accuracy = 0.0
		count = 0
		for i in range(2, VIDEO_LEN - ahead_time, 1):
			yaw_predict_value, yaw_predict_quan = uti.predict_yaw_trun(yaw_trace, i - ALPHA_AHEAD, i+ahead_time)
			pitch_predict_value, pitch_predict_quan = uti.predict_pitch_trun(pitch_trace, i - ALPHA_AHEAD, i+ahead_time)

			start_frame_index = (i + ahead_time)*VIDEO_FPS
			end_frame_index = (i + ahead_time + 1)*VIDEO_FPS
			real_yaw_trace = yaw_trace[start_frame_index:end_frame_index]
			real_pitch_trace = pitch_trace[start_frame_index:end_frame_index]


			el_accuracy = uti.cal_accuracy(yaw_predict_value, yaw_predict_quan, pitch_predict_value, pitch_predict_quan, real_yaw_trace, real_pitch_trace, 1.0)
			sum_accuracy += el_accuracy
			count += 1
		accuracy_list2[ahead_time] = sum_accuracy/count

	print("accuracy_1 is %s" % accuracy_list1)
	print("accuracy_2 is %s" % accuracy_list2)


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

	# yaw_trace_data_1 = [x + 90 for x in yaw_trace_data_1]
	# for i in range(len(yaw_trace_data_1)):
	# 	if yaw_trace_data_1[i] >= 180:
	# 		yaw_trace_data_1[i] -= 360
	# 	if yaw_trace_data_1[i] < -180:
	# 		yaw_trace_data_1[i] += 360

	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in yaw_trace_data_1[:VIDEO_LEN*VIDEO_FPS]], 'o',\
		color='cornflowerblue', markeredgecolor='cornflowerblue', markersize = 2.8, label='FoV Trace 1')  #,linewidth=2.5)
	plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), [x for x in yaw_trace_data_2[:VIDEO_LEN*VIDEO_FPS]],'o',\
		color='chocolate', markeredgecolor='chocolate', markersize = 2.8, label='FoV Trace 2')  #,linewidth=2.5)
	plt.plot(range(1,VIDEO_LEN+1), up_imaginary,'--', color='gray', linewidth=1.5)
	plt.plot(range(1,VIDEO_LEN+1), low_imaginary,'--', color='gray', linewidth=1.5)
	# plt.plot(range(1,VIDEO_LEN+1), zero_line,'--', color='gray', linewidth=1.5)
	plt.legend(loc='upper right', fontsize = 22, markerscale=3, ncol=2)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Degree', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, -240 , 300])
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
	plt.legend(loc='upper right', fontsize = 22, markerscale=3, ncol=2)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Degree', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, -120 , 150])
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
		if not REVISION:
			h.savefig('./figures/fov/yaw.eps', format='eps', dpi=1000, figsize=(30, 10))
			i.savefig('./figures/fov/pitch.eps', format='eps', dpi=1000, figsize=(30, 10))
		# j.savefig('./figures/fov/fov_2_yaw.eps', format='eps', dpi=1000, figsize=(30, 10))
		# k.savefig('./figures/fov/fov_2_pitch.eps', format='eps', dpi=1000, figsize=(30, 10))
		else:
			h.savefig('./figures/fov/yaw_revision.eps', format='eps', dpi=1000, figsize=(30, 10))
			i.savefig('./figures/fov/pitch_revision.eps', format='eps', dpi=1000, figsize=(30, 10))

if __name__ == '__main__':
	main()
