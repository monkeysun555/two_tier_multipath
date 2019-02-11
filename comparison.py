# Utilities for 5G dynamic related simulations
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle 


# Video info
VIDEO_LEN = 300
CHUNK_DURATION = 1.0

# For alpha/gamma
# ALPHA_CAL_LEN is the length of calculating alpha, in seconds
# GAMMA_CAL_LEN is the length of calculating gamma, in seconds
ALPHA_DYNAMIC = 1	# <======================= alpha control
IS_NON_LAYERED = 1  # <======================= whether it is non-layered coding
IS_SAVING = 0		# for non-dynamic. set to zero
IS_SAVING_STATIC = 0	# For fov 2, disable all saving
BUFFER_RANGE = 4
ALPHA_CAL_LEN = 5
MIN_ALPHA_CAL_LEN = 5
GAMMA_CAL_LEN = 10
BW_THRESHOLD = 0.1
STD_THRESHOLD = 0.1
ALPHA_AHEAD = 1.0

ALPHA_CURVE = [[0.966, 0.929, 0.882, 0.834],\
				[0.877, 0.786, 0.707, 0.637],\
				0.950, 0.910, 0.870, 0.810]

GAMMA_CURVE = [[0.956, 1.0, 1.0, 1.0],\
			   [0.883, 0.987, 1.0, 1.0],\
			   [0.823, 0.942, 0.978, 1.0],\
			   [0.780, 0.913, 0.970, 1.0],\
			   [0.669, 0.797, 0.862, 0.893],\

				# For static using 150s (first phase), and all other dynamic
			   [0.823, 0.942, 0.978, 1.0],\


				# Only for static using total trace to do optimization
			   # [0.888, 0.899, 0.923, 0.945],\

			   # For benchmark hmm traces
			   [0.883, 0.987, 1.0, 1.0]]

BT_RATES = [70, 150]
ET_RATES = [280, 440, 600]
# Rate Allocation
BITRATE_LEN = 4
INIT_BL_RATIO = 0.3
INIT_EL_RATIO = 0.7
EL_LOWEST_RATIO = 0.8
EL_HIGHER_RATIO = 1.2
BW_UTI_RATIO = 0.85
BW_DALAY_RATIO = 0.95

# BW prediction
BW_PRED_SAMPLE_SIZE = 10	# second
INIT_BW = 500

# FOV prediction
VIDEO_FPS = 30
VIEW_PRED_SAMPLE_LEN = 30
FRAME_MV_LIMIT = 180.0
POLY_ORDER = 1

ET_HOR_SPAN = 150.0
ET_VER_SPAN = 150.0
BT_HOR_SPAN = 360.0
BT_VER_SPAN = 180.0
VP_HOR_SPAN = 120.0
VP_VER_SPAN = 120.0
VP_BT_RATIO = (VP_HOR_SPAN*VP_VER_SPAN)/(BT_HOR_SPAN*BT_VER_SPAN)
VP_ET_RATIO = (VP_HOR_SPAN*VP_VER_SPAN)/(ET_HOR_SPAN*ET_VER_SPAN)

# Streaming Quality
BUFFER_BL_INIT = 10
BUFFER_EL_INIT = 1

Q_a = 4.91
Q_b = 1.89
Q_c = -1
Q_d = -1

Q_a_new = -1.518
Q_b_new = 1.89
Q_c_new = -1
Q_d_new = -1
# Plot info
FIGURE_NUM = 1
SHOW_FIG = 1


def cmp_load_init_rates(average_bw, video_file, fov_file, coding_type = 2, calculate_gamma = True, buffer_setting = 1):
	if not calculate_gamma:
		rate_cut = [0.0] * BITRATE_LEN

		rate_cut[0] = INIT_BL_RATIO * BW_UTI_RATIO * average_bw
		rate_cut[1] = INIT_EL_RATIO * EL_LOWEST_RATIO * BW_UTI_RATIO * average_bw
		rate_cut[2] = INIT_EL_RATIO * BW_UTI_RATIO * average_bw
		rate_cut[3] = INIT_EL_RATIO * EL_HIGHER_RATIO * BW_UTI_RATIO * average_bw

		# For generating gamma
		# rate_cut[0] = 120
		# rate_cut[1] = 350
		# rate_cut[2] = 450
		# rate_cut[3] = 550

		# print(rate_cut)
		optimal_buffer_len = buffer_setting
		alpha_index = -1
		gamma_index = -1
	else:
		alpha_index = 0
		gamma_index = 0
		if fov_file == './traces/output/Video_9_alpha_beta_new.mat':
			alpha_index = 0
		elif fov_file == './traces/output/Video_13_alpha_beta_new.mat':
			alpha_index = 1
		else:
			alpha_index = 2

		if video_file == './traces/bandwidth/BW_Trace_5G_0.txt':
			gamma_index = 0
		elif video_file == './traces/bandwidth/BW_Trace_5G_1.txt':
			gamma_index = 1
		elif video_file == './traces/bandwidth/BW_Trace_5G_2.txt':
			gamma_index = 2
		elif video_file == './traces/bandwidth/BW_Trace_5G_3.txt':
			gamma_index = 3
		elif video_file == './traces/bandwidth/BW_Trace_5G_4.txt':
			gamma_index = 4
		elif video_file == './traces/bandwidth/BW_Trace_5G_5.txt':
			gamma_index = 5
		else:
			gamma_index = 6
		
		alpha_curve = ALPHA_CURVE[alpha_index]
		gamma_curve = GAMMA_CURVE[gamma_index]

		# rate_cut, optimal_buffer_len = calculate_rate_cute_non_layer(average_bw, alpha_curve, gamma_curve, coding_type)

		rate_cut = [BT_RATES[0], ET_RATES[0], ET_RATES[1], ET_RATES[2]]
		optimal_buffer_len = 2
	return rate_cut, optimal_buffer_len, alpha_index, gamma_index


# 1 for layered_coding
# 2 for non-layered coding
def cmp_rate_optimize(display_time, average_bw, std_bw, alpha_history, version, fov_file, coding_type = 2):
	gamma_curve = update_gamma(average_bw, std_bw)
	alpha_curve = update_alpha(display_time, alpha_history, version, fov_file)
	# print(alpha_curve)
	alpha_gamma = np.multiply(alpha_curve, gamma_curve)
	optimal_alpha_gamma = np.amax(alpha_gamma)
	optimal_buffer_len = np.argmax(alpha_gamma)

	if coding_type == 1:
		beta = ((1 - optimal_alpha_gamma)*(ET_VER_SPAN*ET_HOR_SPAN))/ \
				(optimal_alpha_gamma*(BT_VER_SPAN*BT_HOR_SPAN - ET_HOR_SPAN*ET_VER_SPAN))

		rate_et_average = BW_UTI_RATIO*average_bw/ \
						(1 + (BT_HOR_SPAN*BT_VER_SPAN*beta)/((1-beta)*ET_HOR_SPAN*ET_VER_SPAN))
	elif coding_type == 2:
		beta = ((1 - optimal_alpha_gamma)*(ET_VER_SPAN*ET_HOR_SPAN))/ \
				(optimal_alpha_gamma*BT_VER_SPAN*BT_HOR_SPAN)
		rate_et_average = BW_UTI_RATIO*average_bw/ \
				(1 + ((BT_HOR_SPAN*BT_VER_SPAN*beta)/(ET_HOR_SPAN*ET_VER_SPAN)))

	rate_bt_average = BW_UTI_RATIO*average_bw - rate_et_average
	rate_et_low = rate_et_average * EL_LOWEST_RATIO
	rate_et_high = rate_et_average * EL_HIGHER_RATIO

	new_bt = cmp_quantize_bt_rate(rate_bt_average)

	rate_cuts = [new_bt, ET_RATES[0], ET_RATES[1], ET_RATES[2]]
	# print(optimal_buffer_len)

	#update rate cut and rate cut version
	# print("Do a optimzation, current time is %s" % display_time)
	# print("Alpha is: ", alpha_curve)
	# print("Gamma is: ", gamma_curve)
	# print("alpha gamma is: ", alpha_gamma)
	# print("average bw is %s, std is %s, rate_bt is: %s and rate_et is: %s" %(average_bw, std_bw, rate_bt_average, rate_et_average))
	# print("Rate cute is: %s" % rate_cuts)
	# print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	# print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	# return [rate_bt_average, rate_et_low, rate_et_average, rate_et_high], optimal_buffer_len
	return rate_cuts, optimal_buffer_len

def cmp_quantize_bt_rate(bt):
	# if bt < BT_RATES[0] or bt > BT_RATES[-1] or et2 < ET_RATES[0] or et2 > ET_RATES[-1]:
	# 	print("out of range")
	bt_budget = 0
	bt_array = np.asarray(BT_RATES)
	et_array = np.asarray(ET_RATES)
	idx_bt = (np.abs(bt_array - bt)).argmin()
	bt_budget = bt - BT_RATES[idx_bt]

	return BT_RATES[idx_bt]

def cmp_load_viewport(vp_trace, video_length):
	# mat_contents = sio.loadmat(vp_trace)

	fov_traces = pickle.load(open(vp_trace, "rb"))
	yaw_trace_data = (fov_traces['gt_theta']/math.pi)*180.0 + 180.0
	pitch_trace_data = (fov_traces['gt_phi']/math.pi)*180.0 + 90.0

	for i in range(len(yaw_trace_data)):
		if math.isnan(yaw_trace_data[i]) or math.isnan(pitch_trace_data[i]):
			del yaw_trace_data[i]
			del pitch_trace_data[i]
	# if vp_trace == ''

	assert len(yaw_trace_data) == len(pitch_trace_data)

	return yaw_trace_data[:video_length*VIDEO_FPS], pitch_trace_data[:video_length*VIDEO_FPS]

def update_gamma(average_bw, std_bw):
	# Fix gamma
	current_std_mean_ratio = std_bw/average_bw
	std_mean_ratios = [22.51/734.34, 85.92/722.80, 154.81/719.03, 206.73/659.67, 277.10/585.31]
	std_mean_array = np.asarray(std_mean_ratios)
	idx = (np.abs(std_mean_array - current_std_mean_ratio)).argmin()
	return GAMMA_CURVE[idx]

def update_alpha(display_time, alpha_history, version, fov_file, buffer_range = BUFFER_RANGE):
	alpha = [0.0]*buffer_range
	if ALPHA_DYNAMIC:
		alpha_calculation_len = 0
		for i in reversed(range(len(alpha_history)-1)):
			if alpha_history[i][2] != version and alpha_history[i][2] != version - 1 and alpha_history[i][2] != version - 2:  # COMMENT IF gamma is 15s, 
				break
			alpha_calculation_len += 1	

		alpha_calculation_len = np.minimum(alpha_calculation_len, ALPHA_CAL_LEN)
		# alpha_calculation_len = np.minimum(len(alpha_history)-BUFFER_RANGE, ALPHA_CAL_LEN)
		# print("alpha_history is: ", alpha_history, alpha_calculation_len)
		assert alpha_calculation_len >= 1
		# print(alpha_calculation_len)
		if version > 0 and alpha_calculation_len >= MIN_ALPHA_CAL_LEN:	# 100 means will not triggered; if set to "0", change calculation of alpha_calculation_len to upper one
			# print(alpha_calculation_len)
			# print(alpha_history)
			for i in range(buffer_range):
				temp_alpha = 0.0
				temp_alpha_count = 0
				for j in range(alpha_calculation_len):
					temp_alpha += alpha_history[-(j+i+2)][1][i][4]
					temp_alpha_count += 1
				if temp_alpha_count == 0:
					alpha[i] = 0.0
				else:
					alpha[i] = temp_alpha/temp_alpha_count
		else:
			# Version 1
			# for i in range(buffer_range):
			# 	temp_alpha = 0.0
			# 	temp_alpha_count = 0
			# 	for j in range(alpha_calculation_len-i):
			# 		temp_alpha += alpha_history[-(j+i+2)][1][i][4]
			# 		temp_alpha_count += 1
			# 	if temp_alpha_count == 0:
			# 		alpha[i] = 0.0
			# 	else:
			# 		alpha[i] = temp_alpha/temp_alpha_count

			# Version 2
			if fov_file == './traces/output/Video_9_alpha_beta_new.mat':
				alpha = [0.966, 0.929, 0.882, 0.834]
			elif fov_file == './traces/output/Video_13_alpha_beta_new.mat':
				alpha = [0.877, 0.786, 0.707, 0.637]
			else:
				alpha = [0.966, 0.929, 0.882, 0.834]

	# Statistic alpha curve
	else:
		alpha = [0.911, 0.870, 0.814, 0.771]
	
	return alpha

def get_hitrate(current_time, hitrate, index):
	return np.mean(hitrate[int(current_time)-1][index*VIDEO_FPS:(index+1)*VIDEO_FPS])


def cmp_show_result(streaming, video_length, coding_type):
	# record figures
	# figures = []
	# figures.append([show_buffer(streaming, video_length), 'buffer'])
	# #	display and received bitrates
	# figures.append([show_rates(streaming, video_length, coding_type),'rate'])

	# if streaming.dynamic == 1:
	# 	figures.append([show_rate_cuts(streaming, video_length), 'cuts'])

	total_reward = cmp_show_rates(streaming, video_length, coding_type)

	# figures.append([show_received_rates(streaming, video_length, coding_type),'received_rate'])

	# show_bw(streaming)

	# for i in streaming.alpha_history:
	# 	print(i)
	

	# if SHOW_FIG:
	# 	show_figure(figures)

	# if IS_SAVING:
	# 	if streaming.dynamic == 1:
	# 		if streaming.dy_type == 'fix':
	# 			for fig in figures:
	# 				fig[0].savefig('./figures/fix/fix'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
			
	# 		elif streaming.dy_type == 'adaptive':
	# 			for fig in figures:
	# 				fig[0].savefig('./figures/adaptive/adaptive'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
			
	# 		elif streaming.dy_type == 'std_adaptive':
	# 			for fig in figures:
	# 				fig[0].savefig('./figures/std_adaptive/std_adaptive'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
			
	# 	else:
	# 		if streaming.dy_type == 'fix':
	# 			for fig in figures:
	# 				fig[0].savefig('./figures/static/static'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
	# 		elif streaming.dy_type == 'adaptive':
	# 			print("adaptive is not for static")

	return total_reward


# show results
def cmp_show_rates(streaming, video_length, coding_type = 2):
	# For video rate
	receive_bitrate = [0.0]*video_length
	display_bitrate = [0.0]*video_length
	deliver_bitrate = [0.0]*video_length
	log_bitrate = [0.0]*video_length
	average_frame_quality = [0.0]*video_length
	frame_quality_record = [0]*video_length
	new_quality = [0.0]*video_length
	new_quality_record = [0]*video_length
	total_alpha = 0.0	# fov accuracy ratio
	total_gamma = 0.0	# chunk pass ratio
	bl_info = streaming.evr_bl_recordset
	el_info = streaming.evr_el_recordset
	total_bt_rate = 0.0
	total_et_rate = 0.0
	# <update> later
	rate_cut = streaming.rate_cut
	rebuf = streaming.freezing_time
	# for hit rate
	average_frame_quality_hit = [0.0]*video_length
	average_frame_quality_hit_raw =[0.0]*video_length
	# Layered coding
	if coding_type == 1:
		for i in range(BUFFER_BL_INIT):
			display_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]
			receive_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]
			deliver_bitrate[i] += rate_cut[0][0]

		for i in range(BUFFER_EL_INIT):
			display_bitrate[i] += VP_ET_RATIO * rate_cut[0][-1]
			receive_bitrate[i] += VP_ET_RATIO * rate_cut[0][-1]
			deliver_bitrate[i] += rate_cut[0][-1]


		for i in range(len(bl_info)):
			display_bitrate[bl_info[i][0]] += VP_BT_RATIO * rate_cut[bl_info[i][5]][bl_info[i][1]]
			receive_bitrate[bl_info[i][0]] += VP_BT_RATIO * rate_cut[bl_info[i][5]][bl_info[i][1]]
			deliver_bitrate[bl_info[i][0]] += rate_cut[bl_info[i][5]][bl_info[i][1]]

		for i in range(len(el_info)):
			time_eff = el_info[i][9]

			start_frame_index = int((1 - time_eff + el_info[i][0])*VIDEO_FPS)
			end_frame_index = int((1 + el_info[i][0])*VIDEO_FPS)

			quan_yaw = el_info[i][5]
			real_yaw = el_info[i][6]
			real_yaw_trace = streaming.yaw_trace[start_frame_index:end_frame_index]
			quan_pitch = el_info[i][7]
			real_pitch = el_info[i][8]
			real_pitch_trace = streaming.pitch_trace[start_frame_index:end_frame_index]

			el_accuracy = cal_accuracy(real_yaw, quan_yaw, real_pitch, quan_pitch, real_yaw_trace, real_pitch_trace, time_eff)
			if time_eff == 0:
				assert el_accuracy == 0
			receive_bitrate[el_info[i][0]] += VP_ET_RATIO * rate_cut[el_info[i][10]][el_info[i][1]]
			display_bitrate[el_info[i][0]] += VP_ET_RATIO * time_eff * el_accuracy * rate_cut[el_info[i][10]][el_info[i][1]]
			deliver_bitrate[el_info[i][0]] += rate_cut[el_info[i][10]][el_info[i][1]]

			total_alpha += el_accuracy
			total_gamma += time_eff

	# Non-layered coding
	elif coding_type == 2:
		for i in range(BUFFER_BL_INIT):
			display_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]
			receive_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]
			deliver_bitrate[i] += rate_cut[0][0]
			total_bt_rate += rate_cut[0][0]

		for i in range(BUFFER_EL_INIT):
			display_bitrate[i] = VP_ET_RATIO * rate_cut[0][-1]
			receive_bitrate[i] = VP_ET_RATIO * rate_cut[0][-1]
			deliver_bitrate[i] += rate_cut[0][-1]
			average_frame_quality[i] += get_quality((VP_ET_RATIO * rate_cut[0][-1])/VIDEO_FPS, 0.0, 0.0)
			average_frame_quality_hit[i] += get_quality((VP_ET_RATIO * rate_cut[0][-1])/VIDEO_FPS, 0.0, 0.0)
			average_frame_quality_hit_raw[i] += get_quality((VP_ET_RATIO * rate_cut[0][-1])/VIDEO_FPS, 0.0, 0.0)
			# print(average_frame_quality)
			new_quality[i] += new_get_quality(VP_ET_RATIO * rate_cut[0][-1], 0.0, 0.0)
			frame_quality_record[i] = 1
			new_quality_record[i] = 1
			total_et_rate += rate_cut[0][-1]

		for i in range(len(bl_info)):
			display_bitrate[bl_info[i][0]] += VP_BT_RATIO * rate_cut[bl_info[i][5]][bl_info[i][1]]
			receive_bitrate[bl_info[i][0]] += VP_BT_RATIO * rate_cut[bl_info[i][5]][bl_info[i][1]]
			deliver_bitrate[bl_info[i][0]] += rate_cut[bl_info[i][5]][bl_info[i][1]]
			total_bt_rate += rate_cut[bl_info[i][5]][bl_info[i][1]]

		for i in range(len(el_info)):
			time_eff = el_info[i][9]
			start_frame_index = int((1 - time_eff + el_info[i][0])*VIDEO_FPS)
			end_frame_index = int((1 + el_info[i][0])*VIDEO_FPS)

			quan_yaw = el_info[i][5]
			real_yaw = el_info[i][6]
			real_yaw_trace = streaming.yaw_trace[start_frame_index:end_frame_index]
			quan_pitch = el_info[i][7]
			real_pitch = el_info[i][8]
			real_pitch_trace = streaming.pitch_trace[start_frame_index:end_frame_index]

			# This if for TLP prediction accuracy calculation
			el_accuracy = cal_accuracy(real_yaw, quan_yaw, real_pitch, quan_pitch, real_yaw_trace, real_pitch_trace, time_eff)

			#################################
			# This is for using hitrate curve
			chunk_hit = el_info[i][11]
			effec_hit = []
			hit_el_accuracy = 0.0
			if time_eff != 0:
				effec_hit = chunk_hit[int((1-time_eff)*VIDEO_FPS):VIDEO_FPS]
				hit_el_accuracy = np.mean(effec_hit)

			average_frame_quality[el_info[i][0]] = frame_nlc_quality(quan_yaw, quan_pitch, real_yaw_trace, real_pitch_trace, time_eff, \
									display_bitrate[el_info[i][0]], VP_ET_RATIO * rate_cut[el_info[i][10]][el_info[i][1]])

			average_frame_quality_hit[el_info[i][0]] = frame_nlc_quality_hit(effec_hit, time_eff, display_bitrate[el_info[i][0]], VP_ET_RATIO * rate_cut[el_info[i][10]][el_info[i][1]])

			# For hit rate raw calculation
			average_frame_quality_hit_raw[el_info[i][0]] = Q_a_new + Q_b_new * np.log(VP_ET_RATIO * rate_cut[el_info[i][10]][el_info[i][1]])
			#################################

			frame_quality_record[el_info[i][0]] = 1

			if VP_ET_RATIO * time_eff * el_accuracy * rate_cut[el_info[i][10]][el_info[i][1]] == 0:
				new_quality[el_info[i][0]] += new_get_quality(VP_BT_RATIO * rate_cut[el_info[i][10]][0], 0.0, 0.0)
				new_quality_record[el_info[i][0]] = 1
			else:
				# et_existing_ratio = time_eff * el_accuracy
				# assert et_existing_ratio > 0
				# frame_level_eff = len(real_yaw_trace)/VIDEO_FPS
				new_quality[el_info[i][0]] += (1-time_eff) * new_get_quality(display_bitrate[el_info[i][0]], 0.0, 0.0)
				new_quality[el_info[i][0]] += time_eff * el_accuracy*new_get_quality(VP_ET_RATIO * rate_cut[el_info[i][10]][el_info[i][1]], 0.0, 0.0)
				new_quality[el_info[i][0]] += time_eff * (1-el_accuracy)*new_get_quality(display_bitrate[el_info[i][0]], 0.0, 0.0)
				new_quality_record[el_info[i][0]] = 1

			if time_eff == 0:
				assert el_accuracy == 0

			receive_bitrate[el_info[i][0]] = 0
			receive_bitrate[el_info[i][0]] += VP_ET_RATIO * rate_cut[el_info[i][10]][el_info[i][1]]


			display_bitrate[el_info[i][0]] -= time_eff * el_accuracy * display_bitrate[el_info[i][0]]
			display_bitrate[el_info[i][0]] += VP_ET_RATIO * time_eff * el_accuracy * rate_cut[el_info[i][10]][el_info[i][1]]

			deliver_bitrate[el_info[i][0]] += rate_cut[el_info[i][10]][el_info[i][1]]
			total_et_rate += rate_cut[el_info[i][10]][el_info[i][1]]

			total_alpha += el_accuracy
			total_gamma += time_eff

	for i in range(len(average_frame_quality)):
		if frame_quality_record[i] == 0:
			average_frame_quality[i] += get_quality(display_bitrate[i]/VIDEO_FPS, 0.0, 0.0)
	######################################
	# for hit rate
	for i in range(len(average_frame_quality_hit)):
		if frame_quality_record[i] == 0:
			average_frame_quality_hit[i] += get_quality(display_bitrate[i]/VIDEO_FPS, 0.0, 0.0)

	for i in range(len(average_frame_quality_hit_raw)):
		if frame_quality_record[i] == 0:
			average_frame_quality_hit_raw[i] += get_quality(display_bitrate[i]/VIDEO_FPS, 0.0, 0.0)
	######################################

	for i in range(len(new_quality_record)):
		if new_quality_record[i] == 0:
			new_quality[i] += new_get_quality(display_bitrate[i], 0.0, 0.0)

	for i in range(len(display_bitrate)):	# to QoE
		if display_bitrate[i] == 0.0:
			print("cannot be like this at %s" % i)
			continue
		log_bitrate[i] += get_quality(display_bitrate[i], 0.0, 0.0)


	# print("Fake alpha ratio: ", (total_alpha + 1.0)/video_length)
	# print("Average effective alpha: ", (total_alpha + 1.0)/(len(el_info) + 1))
	# print("EL existing ratio: ", (len(el_info)+1.0)/video_length)
	# print("Average gamma: ", (total_gamma + 1.0)/video_length)
	# print("Average BT rates", total_bt_rate/video_length)
	# print("Average ET rates", total_et_rate/video_length)
	# # print("Displayed effective rate sum: ", sum(display_bitrate))
	# # print("Received effective rate sum: ", sum(receive_bitrate))
	# # print("Log rate sum: ", sum(log_bitrate))
	# print("<===========================================>")
	# # print(video_length)
	# print("Average Quality (wrong): ", sum(log_bitrate)/video_length)	 ##
	# print("Averate Per Frame Quality (right): ", sum(average_frame_quality)/video_length)
	# print("Average Quality (new rate): ", sum(new_quality)/video_length)		
	# print("Displayed rate sum: ", sum(display_bitrate))
	# print("Average Displayed rate: ", sum(display_bitrate)/video_length)	
	# print("Received rate sum: ", sum(deliver_bitrate))
	# print("Average received rate: ", sum(deliver_bitrate)/video_length)	
	# print("Total rebuffering is: %s" % rebuf)
	# print("Number of optimization: %s" % (streaming.rate_cut_version + 1))
	# print("<=================================>")
	# print("Rate cut info is as %s" % streaming.rate_cut)
	# print("Rate cut version is as %s" % streaming.rate_cut_version)
	# print("Rate cut time is as %s" % streaming.rate_cut_time)



	# global FIGURE_NUM
	# g = plt.figure(FIGURE_NUM,figsize=(20,5))
	# FIGURE_NUM += 1
	# plt.plot(range(1,video_length+1), display_bitrate, 'gv-', markersize = 3,\
	#  markeredgewidth = 0.1, markeredgecolor = 'green', linewidth = 2, label='Rendered Bitrate')
	# plt.plot(range(1,video_length+1), receive_bitrate, 'ro-', markersize = 4, \
	#  markeredgewidth = 0.05, markeredgecolor = 'red', linewidth = 1, label='Received Effective Bitrate')
	# plt.plot(range(1,video_length+1), deliver_bitrate, 'b*-', markersize = 5, \
	#  markeredgewidth = 0.2, markeredgecolor = 'blue', linewidth = 2, label='Received Bitrate')
	# plt.legend(loc='upper right', fontsize=22, ncol=3, mode='expand')
	# # plt.title('Effective & Received Video Bitrate')
	# plt.xlabel('Second', fontsize =30)
	# plt.ylabel('Mbps',fontsize = 30)
	# plt.tick_params(axis='both', which='major', labelsize=30)
	# plt.tick_params(axis='both', which='minor', labelsize=30)
	# plt.xticks(np.arange(0, video_length+1, 50))
	# plt.yticks(np.arange(200, 1201, 200))
	# plt.gcf().subplots_adjust(bottom=0.20, left=0.1, right=0.97)	
	# plt.axis([0, video_length, 0, 1200])

	# return sum(average_frame_quality) 
	######################################
	# For hitrate
	return sum(average_frame_quality_hit)		## Right
	# return sum(average_frame_quality_hit_raw)	## Wrong, upper bound for PI without accuracy
	######################################

def cal_accuracy(pred_yaw_value, pred_yaw_quan, pred_pitch_value, pred_pitch_quan, real_yaw_trace, real_pitch_trace, time_eff, pred_type = 'quan'):
	if len(real_yaw_trace) == 0:
		assert time_eff == 0
		return 0.0
	# For yaw
	if pred_type == 'quan':
		pred_yaw_value = pred_yaw_quan * 30.0 + 15.0
		pred_pitch_value = pred_pitch_quan * 30.0

	sum_yaw_eff = 0.0
	yaw_eff = 0.0
	for i in range(len(real_yaw_trace)):
		yaw_distance = np.minimum(np.abs(real_yaw_trace[i] - pred_yaw_value), 360.0 - np.abs(real_yaw_trace[i] - pred_yaw_value)) 
		# percentage of VP, not ET, inlcuding ET vidoe content
		yaw_accuracy = np.minimum(1.0, np.maximum(0.0, ((VP_HOR_SPAN + ET_HOR_SPAN)/2.0 - yaw_distance)/VP_HOR_SPAN))	
		sum_yaw_eff += yaw_accuracy

	yaw_eff = sum_yaw_eff/len(real_yaw_trace)

	sum_pitch_eff = 0.0
	pitch_eff = 0.0
	# For pitch
	for i in range(len(real_pitch_trace)):
		pitch_distance = np.abs(pred_pitch_value - real_pitch_trace[i])
		pitch_accuracy = np.minimum(1.0, np.maximum(0.0, (VP_VER_SPAN + ET_VER_SPAN)/2.0 - pitch_distance)/VP_VER_SPAN)
		sum_pitch_eff += pitch_accuracy
	pitch_eff = sum_pitch_eff/len(real_pitch_trace)

	area_accuracy = yaw_eff * pitch_eff
	return area_accuracy


def frame_nlc_quality_hit(real_hit, time_eff, base_rate, et_rate):
	if len(real_hit) == 0:
		assert time_eff == 0
		return get_quality(base_rate/VIDEO_FPS, 0.0, 0.0)
	
	base_frame_length = VIDEO_FPS - len(real_hit)
	quality_without_et = base_frame_length * get_quality(base_rate/VIDEO_FPS, 0.0, 0.0)

	quality_with_et = 0.0
	for i in range(len(real_hit)):
		area_accuracy = real_hit[i]
		quality_with_et_bt = 0.0
		quality_with_et_et = 0.0
		# area_accuracy = 1

		if area_accuracy != 0 and area_accuracy != 1:
			eff_frame_rate_bt = base_rate/VIDEO_FPS
			eff_frame_rate_et = et_rate/VIDEO_FPS
			# print(area_accuracy, base_rate/VIDEO_FPS, et_rate/VIDEO_FPS)
			quality_with_et_bt = get_quality(eff_frame_rate_bt, 0.0, 0.0) * (1.0 - area_accuracy)
			quality_with_et_et = get_quality(eff_frame_rate_et, 0.0, 0.0) * area_accuracy
			quality_with_et += quality_with_et_bt
			quality_with_et += quality_with_et_et

		elif area_accuracy == 0:
			eff_frame_rate_bt = base_rate/VIDEO_FPS			
			quality_with_et_bt = get_quality(eff_frame_rate_bt, 0.0, 0.0)
			quality_with_et += quality_with_et_bt	

		elif area_accuracy == 1:					
			eff_frame_rate_et = et_rate/VIDEO_FPS 
			quality_with_et_et = get_quality(eff_frame_rate_et, 0.0, 0.0)
			quality_with_et += quality_with_et_et
		else:
			print("impossible to happen!")
		
	per_frame_quality = ((quality_with_et + quality_without_et)/VIDEO_FPS)
	# print(per_frame_quality)
	return per_frame_quality


def frame_nlc_quality(quan_yaw, quan_pitch, real_yaw_trace, real_pitch_trace, time_eff, base_rate, et_rate):
	if len(real_yaw_trace) == 0:
		assert time_eff == 0
		return get_quality(base_rate/VIDEO_FPS, 0.0, 0.0)
	# For yaw
	pred_yaw_value = quan_yaw * 30.0 + 15.0
	pred_pitch_value = quan_pitch * 30.0

	yaw_eff = 0.0
	pitch_eff = 0.0

	base_frame_length = VIDEO_FPS - len(real_yaw_trace)
	quality_without_et = base_frame_length * get_quality(base_rate/VIDEO_FPS, 0.0, 0.0)


	quality_with_et = 0.0
	for i in range(len(real_yaw_trace)):
		yaw_distance = np.minimum(np.abs(real_yaw_trace[i] - pred_yaw_value), 360.0 - np.abs(real_yaw_trace[i] - pred_yaw_value)) 
		yaw_accuracy = np.minimum(1.0, np.maximum(0.0, ((VP_HOR_SPAN + ET_HOR_SPAN)/2.0 - yaw_distance)/VP_HOR_SPAN))	
		pitch_distance = np.abs(pred_pitch_value - real_pitch_trace[i])
		pitch_accuracy = np.minimum(1.0, np.maximum(0.0, (VP_VER_SPAN + ET_VER_SPAN)/2.0 - pitch_distance)/VP_VER_SPAN)
		area_accuracy = yaw_accuracy * pitch_accuracy
		quality_with_et_bt = 0.0
		quality_with_et_et = 0.0
		# area_accuracy = 1

		if area_accuracy != 0 and area_accuracy != 1:
			eff_frame_rate_bt = base_rate/VIDEO_FPS
			eff_frame_rate_et = et_rate/VIDEO_FPS
			# print(area_accuracy, base_rate/VIDEO_FPS, et_rate/VIDEO_FPS)
			quality_with_et_bt = get_quality(eff_frame_rate_bt, 0.0, 0.0) * (1.0 - area_accuracy)
			quality_with_et_et = get_quality(eff_frame_rate_et, 0.0, 0.0) * area_accuracy
			quality_with_et += quality_with_et_bt
			quality_with_et += quality_with_et_et

		elif area_accuracy == 0:
			eff_frame_rate_bt = base_rate/VIDEO_FPS			
			quality_with_et_bt = get_quality(eff_frame_rate_bt, 0.0, 0.0)
			quality_with_et += quality_with_et_bt	

		elif area_accuracy == 1:					
			eff_frame_rate_et = et_rate/VIDEO_FPS 
			quality_with_et_et = get_quality(eff_frame_rate_et, 0.0, 0.0)
			quality_with_et += quality_with_et_et
		else:
			print("impossible to happen!")
		
	per_frame_quality = ((quality_with_et + quality_without_et)/VIDEO_FPS)
	# print(per_frame_quality)
	return per_frame_quality


def new_get_quality(eff_rate, rebuff, black, zero_rate = False):
	if not zero_rate:
		quality = Q_a_new + Q_b_new * np.log(eff_rate) + Q_c_new*rebuff + Q_d*black
	else:
		quality = Q_c_new*rebuff + Q_d_new*black
	return quality

def get_quality(eff_rate, rebuff, black, zero_rate = False):
	if not zero_rate:
		quality = Q_a + Q_b * np.log(eff_rate) + Q_c*rebuff  + Q_d*black
	else:
		quality = Q_c*rebuff + Q_d*black
		# quality = 0.0
	return quality