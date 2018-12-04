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
IS_SAVING_STATIC = 0	# To save naive 360, FOV only benchmarks, not test on fov 2, disable for FoV 2, could used for USER 0 but NOT for 6 (revision)
REVISION = 1
BUFFER_RANGE = 4
ALPHA_CAL_LEN = 30
MIN_ALPHA_CAL_LEN = 30
GAMMA_CAL_LEN = 10
BW_THRESHOLD = 0.1
STD_THRESHOLD = 0.1
ALPHA_AHEAD = 1.0
# FIX_INIT = 0	# to control initial setting, for static two tier, disable this value to do optimization. For dynamic, use this for fix initial

# alpha curve: FOV 1, FOV 2 (in previous version,) and then USER 0, USER 6 (for revision)
ALPHA_CURVE = [[0.966, 0.929, 0.882, 0.834],\
				[0.877, 0.786, 0.707, 0.637],\
				[0.933, 0.878, 0.836, 0.800],\
				[0.857, 0.776, 0.713, 0.671]]

GAMMA_CURVE = [[0.956, 1.0, 1.0, 1.0],\
			   [0.883, 0.987, 1.0, 1.0],\
			   [0.823, 0.942, 0.978, 1.0],\
			   [0.780, 0.913, 0.970, 1.0],\
			   [0.669, 0.797, 0.862, 0.893],\

				# Only for static using 150s (first phase), and all other dynamic
			   # [0.823, 0.942, 0.978, 1.0],\


				# Only for static using 450s total trace to do optimization
			   [0.888, 0.899, 0.923, 0.945],\

			   # For benchmark hmm traces
			   [0.883, 0.987, 1.0, 1.0]]

BT_RATES = [10, 30, 50, 80, 120, 160, 200]
ET_RATES = [300, 350, 400, 450, 500, 550, 600, 650, 700]
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
Q_c = -2	# CHANGed for REVISION -2
Q_d = -2	# CHANGed for REVISION -2

Q_a_new = -1.518
Q_b_new = 1.89
Q_c_new = -2	# CHANGed for REVISION -2
Q_d_new = -2	# CHANGed for REVISION -2
# Plot info
FIGURE_NUM = 1
SHOW_FIG = 1

# 1 for layered_coding
# 2 for non-layered coding
def rate_optimize(display_time, average_bw, std_bw, alpha_history, version, fov_file, coding_type = 2):
	gamma_curve = update_gamma(average_bw, std_bw)
	alpha_curve = update_alpha(display_time, alpha_history, fov_file, version)
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

	rate_cuts = quantize_bt_et_rate(average_bw, rate_bt_average, rate_et_low, rate_et_average, rate_et_high)

	#update rate cut and rate cut version
	print("Do a optimzation, current time is %s" % display_time)
	print("Alpha is: ", alpha_curve)
	print("Gamma is: ", gamma_curve)
	print("alpha gamma is: ", alpha_gamma)
	print("average bw is %s, std is %s, rate_bt is: %s and rate_et is: %s" %(average_bw, std_bw, rate_bt_average, rate_et_average))
	print("Rate cute is: %s" % rate_cuts)
	print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	# return [rate_bt_average, rate_et_low, rate_et_average, rate_et_high], optimal_buffer_len
	return rate_cuts, optimal_buffer_len

def rate_optimize_std(display_time, average_bw, std_bw, alpha_history, version, fov_file, coding_type = 2):
	gamma_curve = update_gamma(average_bw, std_bw)
	alpha_curve = update_alpha(display_time, alpha_history, fov_file, version)
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

	rate_cuts = quantize_bt_et_rate(average_bw, rate_bt_average, rate_et_low, rate_et_average, rate_et_high)

	#update rate cut and rate cut version
	print("Do a optimzation, current time is %s" % display_time)
	print("Alpha is: ", alpha_curve)
	print("Gamma is: ", gamma_curve)
	print("alpha gamma is: ", alpha_gamma)
	print("average bw is %s, std is %s, rate_bt is: %s and rate_et is: %s" %(average_bw, std_bw, rate_bt_average, rate_et_average))
	print("Rate cute is: %s" % rate_cuts)
	print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	# return [rate_bt_average, rate_et_low, rate_et_average, rate_et_high], optimal_buffer_len
	return rate_cuts, optimal_buffer_len


def cal_average_bw(network_time, bw_history, last_ref_time):
	average_gamma_bw = 0.0
	std_gamma_bw = 0.0
	average_gamma_real = 0.0
	if network_time - GAMMA_CAL_LEN < last_ref_time:
		average_gamma_bw = -1.0
		std_gamma_bw = -1.0
		average_gamma_real = -1.0
	else:
		ref_time = network_time - GAMMA_CAL_LEN
		temp_bw = []
		temp_real = []
		for i in reversed(range(len(bw_history))):
			if bw_history[i][1] < ref_time:
				break
			else:
				temp_bw.append(bw_history[i][0])
				temp_real.append(bw_history[i][4])
		average_gamma_bw = np.sum(temp_bw)/len(temp_bw)
		std_gamma_bw = np.std(temp_bw)
		average_gamma_real = np.sum(temp_real)/len(temp_real)
	return average_gamma_bw, std_gamma_bw, average_gamma_real

def is_bw_change(current_bw, ref_bw):
	if current_bw >= (1+BW_THRESHOLD)*ref_bw or current_bw <= (1-BW_THRESHOLD)*ref_bw:
		return True
	else:
		return False

def is_bw_std_change(current_bw, current_std, ref_bw, ref_relative_std):
	relative_std = current_std/current_bw
	if current_bw >= (1+BW_THRESHOLD)*ref_bw or current_bw <= (1-BW_THRESHOLD)*ref_bw \
			or np.abs(relative_std - ref_relative_std) >= STD_THRESHOLD:
		return True
	else:
		return False

def record_alpha(yaw_trace, pitch_trace, display_time, video_length, buffer_range = BUFFER_RANGE):
	temp_alpha_len = np.minimum(buffer_range, video_length - int(round(display_time)))
	temp_alpha = [[0] * 5 for i in range(temp_alpha_len)]
	for i in range(temp_alpha_len):
		temp_yaw_value, temp_yaw_quan = predict_yaw_trun(yaw_trace, display_time - ALPHA_AHEAD, int(round(display_time)) + i)
		temp_pitch_value, temp_pitch_quan = predict_pitch_trun(pitch_trace, display_time - ALPHA_AHEAD, int(round(display_time)) + i)
		temp_alpha[i][0] = temp_yaw_value
		temp_alpha[i][1] = temp_yaw_quan
		temp_alpha[i][2] = temp_pitch_value
		temp_alpha[i][3] = temp_pitch_quan
	return temp_alpha


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
		if version > 0 and alpha_calculation_len >= MIN_ALPHA_CAL_LEN:	# 100 means will not triggered; if set to "0", change calculation of alpha_calculation_len to upper one
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
		alpha = [0.911, 0.870, 0.814, 0.771]
	
	return alpha

def get_average_bw(display_time, bw_history, version):
	bw_record = []
	bw_real = []
	# bw_count = 0
	for i in reversed(range(len(bw_history))):
		if bw_history[i][3] == version:
			bw_record.append(bw_history[i][0])
			bw_real.append(bw_history[i][4])
		else:
			break
	average_bw = np.sum(bw_record)/len(bw_record)
	return average_bw, np.std(bw_record), np.sum(bw_real)/len(bw_real) 


def show_network(network_trace):
	print("5G trace mean:", np.mean(network_trace))
	print("5G trace standard deviation:", np.std(network_trace))
	print("5G trace peak:", np.max(network_trace))
	print("5G trace min:", np.min(network_trace))
	print("5G trace median:", np.median(network_trace))

	# print("5G delay mean:", np.mean(network_delay))
	# print("5G delay standard deviation:", np.std(network_delay))
	# print("5G delay peak:", np.max(network_delay))
	# print("5G delay min:", np.min(network_delay))
	# print("5G delay median:", np.median(network_delay))
	return np.mean(network_trace), np.std(network_trace)

def load_init_rates(average_bw, video_file, fov_file, coding_type = 2, calculate_gamma = True, buffer_setting = 1, user = -1, do_dynamic = 0):
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

		print(rate_cut)
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
		elif fov_file == './traces/output/gt_theta_phi_vid_3.p':
			assert user >= 0
			if user == 0:
				alpha_index = 2
			elif user == 6:
				alpha_index = 3

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

		if not do_dynamic:
			rate_cut, optimal_buffer_len = calculate_rate_cute_non_layer(average_bw, alpha_curve, gamma_curve, coding_type)
		# For dynamic, use fix init
		else:
			rate_cut = [BT_RATES[3], ET_RATES[3], ET_RATES[5], ET_RATES[7]]
			optimal_buffer_len = 1

	return rate_cut, optimal_buffer_len, alpha_index, gamma_index

def generate_video_trace(rate_cut, video_length):
	video_trace0 = rate_cut[0] * CHUNK_DURATION * np.ones(video_length)
	video_trace1 = rate_cut[1] * CHUNK_DURATION * np.ones(video_length)
	video_trace2 = rate_cut[2] * CHUNK_DURATION * np.ones(video_length)
	video_trace3 = rate_cut[3] * CHUNK_DURATION * np.ones(video_length)
	return [video_trace0, video_trace1, video_trace2, video_trace3]

def predict_bw(bw_history):
	if len(bw_history) == 0:
		return INIT_BW
	else:
		if len(bw_history) < BW_PRED_SAMPLE_SIZE:
			return sum(row[0] for row in bw_history)/len(bw_history)
		else:
			return sum(row[0] for row in bw_history[-BW_PRED_SAMPLE_SIZE:])/BW_PRED_SAMPLE_SIZE

def load_viewport(vp_trace, video_length):
	mat_contents = sio.loadmat(vp_trace)
	trace_data = mat_contents['data_alpha_beta'] # array of structures
	# pitch_trace_data = mat_contents['view_angle_pitch_combo'] + 90 # array of structures
	# print(len(yaw_trace_data[0]), len(pitch_trace_data[0]))
	# print(trace_data.T.shape)
	yaw_trace_data = (trace_data.T[1]/math.pi)*180.0 + 180.0
	pitch_trace_data = (trace_data.T[2]/math.pi)*180.0 + 90.0
	# print(yaw_trace_data.shape, yaw_trace_data[:VIDEO_LEN*VIDEO_FPS])
	for i in range(len(yaw_trace_data)):
		if math.isnan(yaw_trace_data[i]) or math.isnan(pitch_trace_data[i]):
			del yaw_trace_data[i]
			del pitch_trace_data[i]
	# if vp_trace == ''

	assert len(yaw_trace_data) == len(pitch_trace_data)
	# yaw_trace_data.tolist().remove(float('nan'))
	# pitch_trace_data.tolist().remove(float('nan'))
	# print(np.array(yaw_trace_data).shape)
	# assert (len(yaw_trace_data[0]) > VIDEO_LEN*VIDEO_FPS and len(pitch_trace_data[0]) > VIDEO_LEN*VIDEO_FPS)
	
	# if vp_trace == 'Video_9_alpha_beta_new.mat':
	# 	yaw_trace_data = [x + 90 for x in yaw_trace_data]
	# 	for i in range(len(yaw_trace_data)):
	# 		if yaw_trace_data[i] >= 180:
	# 			yaw_trace_data[i] -= 360
	# 		assert yaw_trace_data[i] >= -180:
	return yaw_trace_data[:video_length*VIDEO_FPS], pitch_trace_data[:video_length*VIDEO_FPS]

def load_pickle_viewport(vp_trace, video_length, user):
	contents_1 = pickle.load(open(vp_trace, "rb"))[1]
	contents_2 = pickle.load(open(vp_trace, "rb"))[2]
	contents_3 = pickle.load(open(vp_trace, "rb"))[3]

	yaw_vid_1 = (contents_1['gt_theta']/math.pi)*180.0
	pitch_vid_1 = (contents_1['gt_phi']/math.pi)*180.0 - 90.0

	yaw_vid_2 = (contents_2['gt_theta']/math.pi)*180.0
	pitch_vid_2 = (contents_2['gt_phi']/math.pi)*180.0 - 90.0

	yaw_vid_3 = (contents_3['gt_theta']/math.pi)*180.0
	pitch_vid_3 = (contents_3['gt_phi']/math.pi)*180.0 - 90.0

	# print(len(yaw_vid_1[USER_1]), len(yaw_vid_3[USER_1]))
	
	# print(yaw_vid_1[USER_1], len(yaw_vid_1[USER_1]))
	yaw_trace_data = yaw_vid_1[user].tolist() + yaw_vid_2[user].tolist() + yaw_vid_3[user].tolist()
	pitch_trace_data = pitch_vid_1[user].tolist() + pitch_vid_2[user].tolist() + pitch_vid_3[user].tolist()

	return yaw_trace_data[:video_length*VIDEO_FPS], pitch_trace_data[:video_length*VIDEO_FPS]


def predict_yaw_trun(yaw_trace, display_time, video_seg_index):
	yaw_predict_value = 0.0
	yaw_predict_quan = 0
	if display_time < 1:
		yaw_predict_value = 0.0
	else:
		vp_index = np.arange(-VIEW_PRED_SAMPLE_LEN,0) + int(display_time*VIDEO_FPS)
			# print(vp_index)
		vp_value = []
		for index in vp_index:
			vp_value.append(yaw_trace[index])
		# print(vp_value)
		for value in vp_value[1:]:
			if value - vp_value[vp_value.index(value)-1] > FRAME_MV_LIMIT:
				value -= 360.0
			elif vp_value[vp_value.index(value)-1] - value > FRAME_MV_LIMIT:
				value += 360.0
		new_value = [vp_value[-1], vp_value[-2]]
		new_index = [vp_index[-1], vp_index[-2]]
		sign = np.sign(vp_value[-1] - vp_value[-2])
		temp = vp_value[-2]
		sign_index = -3 

		for i in reversed(range(VIEW_PRED_SAMPLE_LEN+sign_index+1)):
			if np.sign(temp - vp_value[i]) == sign:
				new_value.append(vp_value[i])
				new_index.append(vp_index[i])
				temp = vp_value[i]
			else:
				break
		new_value.reverse()
		new_index.reverse()	
		yaw_predict_model = np.polyfit(new_index, new_value, POLY_ORDER)
		yaw_predict_idx = int(video_seg_index*VIDEO_FPS + VIDEO_FPS/2)
		yaw_predict_value = np.round(np.polyval(yaw_predict_model,yaw_predict_idx))
	yaw_predict_value %= 360
	yaw_predict_quan = int(yaw_predict_value/30)
	assert yaw_predict_quan < 12 and yaw_predict_quan >= 0
	return yaw_predict_value, yaw_predict_quan

def predict_pitch_trun(pitch_trace, display_time, video_seg_index):
	pitch_predict_value = 0.0
	pitch_predict_quan = 0
	if display_time < 1:
		pitch_predict_value = 0.0
	else:
		vp_index = np.arange(-VIEW_PRED_SAMPLE_LEN,0) + int(display_time*VIDEO_FPS)
			# print(vp_index)
		vp_value = []
		for index in vp_index:
			vp_value.append(pitch_trace[index])
		# for value in vp_value[1:]:
		# 	if value - vp_value[vp_value.index(value)-1] > FRAME_MV_LIMIT:
		# 		value -= 360.0
		# 	elif vp_value[vp_value.index(value)-1] - value > FRAME_MV_LIMIT:
		# 		value += 360.0
		new_value = [vp_value[-1], vp_value[-2]]
		new_index = [vp_index[-1], vp_index[-2]]
		sign = np.sign(vp_value[-1] - vp_value[-2])
		temp = vp_value[-2]
		sign_index = -3 

		for i in reversed(range(VIEW_PRED_SAMPLE_LEN+sign_index+1)):
			if np.sign(temp - vp_value[i]) == sign:
				new_value.append(vp_value[i])
				new_index.append(vp_index[i])
				temp = vp_value[i]
			else:
				break
		new_value.reverse()
		new_index.reverse()	
		pitch_predict_model = np.polyfit(new_index, new_value, POLY_ORDER)
		pitch_predict_idx = int(video_seg_index*VIDEO_FPS + VIDEO_FPS/2)
		pitch_predict_value = np.round(np.polyval(pitch_predict_model,pitch_predict_idx))
	if pitch_predict_value > 180.0:
		pitch_predict_value = 180.0
	if pitch_predict_value < 0.0:
		pitch_predict_value = 0.0


	## pitch quan range from 0 to 6:   0~15, 15 ~ 45, .... 135~165, 165~180
	if pitch_predict_value < 15:
		pitch_predict_quan = 0
	elif pitch_predict_value >= 165:
		pitch_predict_quan = 6
	else:
		pitch_predict_quan = int((pitch_predict_value + 15)/30)
	assert pitch_predict_quan >=0 and pitch_predict_quan <= 6
	return pitch_predict_value, pitch_predict_quan

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


def generate_360_rate():
	# Previous
	# return [100, 250, 400, 550, 700, 850]
	# Revision
	return [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]


def generate_fov_rate():
	# Previous
	# return [100, 250, 400, 550, 700, 850]
	# Revision
	return [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850]

def quantize_bt_et_rate(ave_bw, bt, et1, et2, et3):
	if bt < BT_RATES[0] or bt > BT_RATES[-1] or et2 < ET_RATES[0] or et2 > ET_RATES[-1]:
		print("out of range")
	bt_budget = 0
	bt_array = np.asarray(BT_RATES)
	et_array = np.asarray(ET_RATES)
	idx_bt = (np.abs(bt_array - bt)).argmin()
	bt_budget = bt - BT_RATES[idx_bt]

	et_ave = et2 + bt_budget
	et_low = et_ave * EL_LOWEST_RATIO
	et_high = et_ave * EL_HIGHER_RATIO

	idx_et2 = (np.abs(et_array - et_ave)).argmin()
	idx_et1 = (np.abs(et_array - et_low)).argmin()
	idx_et3 = (np.abs(et_array - et_high)).argmin()
	if idx_et1 == idx_et2 or idx_et2 == idx_et3:
		print("et out of range, should modify")
	if np.abs(idx_et1 - idx_et2) <= 1 or np.abs(idx_et2 - idx_et3) <= 1:
		print("et idx too close, should modify")

	return [BT_RATES[idx_bt], ET_RATES[idx_et1], ET_RATES[idx_et2], ET_RATES[idx_et3]]

def calculate_rate_cute_non_layer(average_bw, alpha_curve, gamma_curve, coding_type = 2):
	# for non-layered coding
	# Change gamma to get rate optimization

	alpha_gamma = np.multiply(alpha_curve, gamma_curve)
	optimal_alpha_gamma = np.amax(alpha_gamma)
	optimal_buffer_len = np.argmax(alpha_gamma)

	# Layered coding
	if coding_type == 1:
		beta = ((1 - optimal_alpha_gamma)*(ET_VER_SPAN*ET_HOR_SPAN))/ \
				(optimal_alpha_gamma*(BT_VER_SPAN*BT_HOR_SPAN - ET_HOR_SPAN*ET_VER_SPAN))

		rate_et_average = BW_UTI_RATIO*average_bw/ \
						(1 + (BT_HOR_SPAN*BT_VER_SPAN*beta)/((1-beta)*ET_HOR_SPAN*ET_VER_SPAN))

	# Non-layered coding
	elif coding_type == 2:
		beta = ((1 - optimal_alpha_gamma)*(ET_VER_SPAN*ET_HOR_SPAN))/ \
				(optimal_alpha_gamma*BT_VER_SPAN*BT_HOR_SPAN)
		rate_et_average = BW_UTI_RATIO*average_bw/ \
				(1 + ((BT_HOR_SPAN*BT_VER_SPAN*beta)/(ET_HOR_SPAN*ET_VER_SPAN)))

	rate_bt_average = BW_UTI_RATIO*average_bw - rate_et_average
	rate_et_low = rate_et_average * EL_LOWEST_RATIO
	rate_et_high = rate_et_average * EL_HIGHER_RATIO

	rate_cuts = quantize_bt_et_rate(average_bw, rate_bt_average, rate_et_low, rate_et_average, rate_et_high)
	print("Alpha is: ", alpha_curve)
	print("Gamma is: ", gamma_curve)
	print("alpha gamma is: ", alpha_gamma)
	print("optimal buffer length is: %s" %optimal_buffer_len)
	print("Beta is: %s" % beta)
	print("average bw is %s, rate_bt is: %s and rate_et is: %s" %(average_bw, rate_bt_average, rate_et_average))
	print("Rate cute is: %s" % rate_cuts)
	# return [rate_bt_average, rate_et_low, rate_et_average, rate_et_high]
	return rate_cuts, optimal_buffer_len


# show results
def show_rates(streaming, video_length, coding_type = 2):
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

	# Layered coding
	if coding_type == 1:
		for i in range(BUFFER_BL_INIT):
			display_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]
			receive_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]
			deliver_bitrate[i] += rate_cut[0][0]

		for i in range(BUFFER_EL_INIT):
			display_bitrate[i] += VP_ET_RATIO * rate_cut[0][-3]
			receive_bitrate[i] += VP_ET_RATIO * rate_cut[0][-3]
			deliver_bitrate[i] += rate_cut[0][-3]


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

			el_accuracy = cal_accuracy(real_yaw, quan_yaw, real_pitch, quan_pitch, real_yaw_trace, real_pitch_trace, time_eff)

			average_frame_quality[el_info[i][0]] = frame_nlc_quality(quan_yaw, quan_pitch, real_yaw_trace, real_pitch_trace, time_eff, \
									display_bitrate[el_info[i][0]], VP_ET_RATIO * rate_cut[el_info[i][10]][el_info[i][1]])
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
			receive_bitrate[el_info[i][0]] = 0	# If ET is received, get rid of BT contribution
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

	for i in range(len(new_quality_record)):
		if new_quality_record[i] == 0:
			new_quality[i] += new_get_quality(display_bitrate[i], 0.0, 0.0)

	for i in range(len(display_bitrate)):	# to QoE
		if display_bitrate[i] == 0.0:
			print("cannot be like this at %s" % i)
			continue
		log_bitrate[i] += get_quality(display_bitrate[i], 0.0, 0.0)


	print("Fake alpha ratio: ", (total_alpha + 1.0)/video_length)
	print("Average effective alpha: ", (total_alpha + 1.0)/(len(el_info) + 1))
	print("EL existing ratio: ", (len(el_info)+1.0)/video_length)
	print("Average gamma: ", (total_gamma + 1.0)/video_length)
	print("Average BT rates", total_bt_rate/video_length)
	print("Average ET rates", total_et_rate/video_length)
	# print("Displayed effective rate sum: ", sum(display_bitrate))
	# print("Received effective rate sum: ", sum(receive_bitrate))
	# print("Log rate sum: ", sum(log_bitrate))
	print("<===========================================>")
	# print(video_length)
	print("Average Quality (wrong): ", sum(log_bitrate)/video_length)	 ##
	print("Averate Per Frame Quality (right): ", sum(average_frame_quality)/video_length)
	print("Average Quality (new rate): ", sum(new_quality)/video_length)		
	print("Displayed rate sum: ", sum(display_bitrate))
	print("Average Displayed rate: ", sum(display_bitrate)/video_length)	
	print("Received rate sum: ", sum(deliver_bitrate))
	print("Average received rate: ", sum(deliver_bitrate)/video_length)	
	print("Total rebuffering is: %s" % rebuf)
	print("Number of optimization: %s" % (streaming.rate_cut_version + 1))
	print("<=================================>")
	print("Rate cut info is as %s" % streaming.rate_cut)
	print("Rate cut version is as %s" % streaming.rate_cut_version)
	print("Rate cut time is as %s" % streaming.rate_cut_time)



	global FIGURE_NUM
	g = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM += 1
	plt.plot(range(1,video_length+1), display_bitrate, 'gv-', markersize = 3,\
	 markeredgewidth = 0.1, markeredgecolor = 'green', linewidth = 2, label='Rendered Bitrate')
	plt.plot(range(1,video_length+1), receive_bitrate, 'ro-', markersize = 4, \
	 markeredgewidth = 0.05, markeredgecolor = 'red', linewidth = 1, label='Received Effective Bitrate')
	plt.plot(range(1,video_length+1), deliver_bitrate, 'b*-', markersize = 5, \
	 markeredgewidth = 0.2, markeredgecolor = 'blue', linewidth = 2, label='Received Bitrate')
	plt.legend(loc='upper right', fontsize=22, ncol=3, mode='expand')
	# plt.title('Effective & Received Video Bitrate')
	plt.xlabel('Second', fontsize =30)
	plt.ylabel('Mbps',fontsize = 30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, video_length+1, 50))
	plt.yticks(np.arange(200, 1201, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.1, right=0.97)	
	plt.axis([0, video_length, 0, 1200])

	return g

def show_buffer(streaming, video_length):
	## Plot buffer length
	buffer_info = streaming.buffer_history

	# print(len(buffer_info))
	# print(buffer_info)
	if np.ceil(streaming.display_time) < video_length:
		time_to_end = video_length - int(np.ceil(streaming.display_time))
		for i in range(time_to_end):
			buffer_info.append([ np.maximum(buffer_info[-1][0]-1, 0),  np.maximum(buffer_info[-1][1]-1, 0), buffer_info[-1][2]+1])
	# print(buffer_info)
	# print(streaming.display_time)
	# print(len(buffer_info))
	global FIGURE_NUM
	a = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM +=1

	columns = 2
	if len(streaming.rate_cut_time) > 1:
		# Plot target
		columns = 3
		target_info = streaming.target_buffer_history
		count = 0
		next_time = streaming.rate_cut_time[count+1]
		target_number = len(streaming.rate_cut_time)
		target_buffer_plot = [0.0]*video_length


		for i in range(video_length):
			if i >= next_time:
				count += 1
				if count >= target_number - 1:
					next_time = video_length + 100
				else:
					next_time = streaming.rate_cut_time[count+1]
				# plt.plot((i,i), (0, 20), '--', color='gray', linewidth=1.5)
			target_buffer_plot[i] = target_info[count] + 1

		plt.plot(range(1, video_length+1), target_buffer_plot,'b^-', markersize = 4, markeredgewidth = 0.5, \
				linewidth = 2, markeredgecolor = 'blue',  label='ET Target Buffer Length')

	plt.plot(range(1, video_length+1), [row[0] for row in buffer_info],'k*-', markersize = 6, markeredgewidth = 0.5, \
			linewidth = 2, markeredgecolor = 'black',  label='BT Buffer Length')
	plt.plot(range(1, video_length+1), [row[1] for row in buffer_info],'ro-', markersize = 2, markeredgewidth = 0.5, \
			linewidth = 2, markeredgecolor = 'red',  label='ET Buffer Length')


	plt.legend(loc='upper right', fontsize=20, ncol=columns)
	# plt.title('BT/ET Buffer Length', fontsize=30)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Second', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, video_length+1, 50))
	plt.yticks(np.arange(5, 21, 5))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085, right=0.97)	
	plt.axis([0, video_length, 0, 13])
	return a 

def show_rate_cuts(streaming, video_length):
	count = 0
	cuts_number = len(streaming.rate_cut_time)

	next_time = streaming.rate_cut_time[count+1]
	bts = [0.0]*video_length
	et1s = [0.0]*video_length
	et2s = [0.0]*video_length
	et3s = [0.0]*video_length


	global FIGURE_NUM
	a = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM +=1

	for i in range(video_length):
		if i >= next_time:
			count += 1
			if count >= cuts_number - 1:
				next_time = video_length + 100
			else:
				next_time = streaming.rate_cut_time[count+1]
			plt.plot((i,i), (0, 1800), '--', color='gray', linewidth=1.5)
		bts[i] = streaming.rate_cut[count][0]
		et1s[i] = streaming.rate_cut[count][1]
		et2s[i] = streaming.rate_cut[count][2]
		et3s[i] = streaming.rate_cut[count][3]


	plt.plot(range(1, video_length+1), bts,'k-', \
			linewidth = 2, label='BT Rate')

	plt.plot(range(1, video_length+1), et1s,'r-', \
			linewidth = 2, label='Lowest ET Rate')

	plt.plot(range(1, video_length+1), et2s,'b-', \
			linewidth = 2, label='Medium ET Rate')

	plt.plot(range(1, video_length+1), et3s,'g-', \
			linewidth = 2, label='Highest ET Rate')

	plt.legend(loc='upper right', fontsize=22, ncol=4, mode='expand')
	# plt.title('BT/ET Buffer Length', fontsize=30)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, video_length+1, 50))
	plt.yticks(np.arange(200, 1601, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.1, right=0.97)	
	plt.axis([0, video_length, 0, 1000])
	return a 


def show_bw(streaming):
	for i in range(len(streaming.video_bw_history)):
		print("Bw at %s is %s and real is %s.\n" % (streaming.video_bw_history[i][1], \
		streaming.video_bw_history[i][0], streaming.video_bw_history[i][4]))
	

def show_figure(figures):
	for f in figures:
		f[0].show()
	raw_input()

def show_result(streaming, video_length, coding_type):
	# record figures
	figures = []
	figures.append([show_buffer(streaming, video_length), 'buffer'])
	#	display and received bitrates
	figures.append([show_rates(streaming, video_length, coding_type),'rate'])

	if streaming.dynamic == 1:
		figures.append([show_rate_cuts(streaming, video_length), 'cuts'])


	# figures.append([show_received_rates(streaming, video_length, coding_type),'received_rate'])

	# show_bw(streaming)

	# for i in streaming.alpha_history:
	# 	print(i)
	
	if SHOW_FIG:
		show_figure(figures)

	if IS_SAVING:
		if not REVISION:
			if streaming.dynamic == 1:
				if streaming.dy_type == 'fix':
					for fig in figures:
						fig[0].savefig('./figures/fix/fix'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
				
				elif streaming.dy_type == 'adaptive':
					for fig in figures:
						fig[0].savefig('./figures/adaptive/adaptive'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
				
				elif streaming.dy_type == 'std_adaptive':
					for fig in figures:
						fig[0].savefig('./figures/std_adaptive/std_adaptive'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
				
			else:
				if streaming.dy_type == 'fix':
					for fig in figures:
						fig[0].savefig('./figures/static/static'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
				elif streaming.dy_type == 'adaptive':
					print("adaptive is not for static")
		else:
			if streaming.dynamic == 1:
				if streaming.dy_type == 'fix':
					for fig in figures:
						fig[0].savefig('./figures/fix/fix'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'_revision.eps', format='eps', dpi=1000, figsize=(30, 10))
				
				elif streaming.dy_type == 'adaptive':
					for fig in figures:
						fig[0].savefig('./figures/adaptive/adaptive'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'_revision.eps', format='eps', dpi=1000, figsize=(30, 10))
				
				elif streaming.dy_type == 'std_adaptive':
					for fig in figures:
						fig[0].savefig('./figures/std_adaptive/std_adaptive'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'_revision.eps', format='eps', dpi=1000, figsize=(30, 10))
				
			else:
				if streaming.dy_type == 'fix':
					for fig in figures:
						fig[0].savefig('./figures/static/static'+fig[1]+str(streaming.network_file)+'_'+str(streaming.fov_file)+'_revision.eps', format='eps', dpi=1000, figsize=(30, 10))
				elif streaming.dy_type == 'adaptive':
					print("adaptive is not for static")

	return

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

def frame_nlc_fov_quality(quan_yaw, quan_pitch, real_yaw_trace, real_pitch_trace, time_eff, rate):
	assert len(real_yaw_trace) == VIDEO_FPS
	# For yaw
	pred_yaw_value = quan_yaw * 30.0 + 15.0
	pred_pitch_value = quan_pitch * 30.0

	yaw_eff = 0.0
	pitch_eff = 0.0

	quality_sum = 0.0
	for i in range(len(real_yaw_trace)):
		yaw_distance = np.minimum(np.abs(real_yaw_trace[i] - pred_yaw_value), 360.0 - np.abs(real_yaw_trace[i] - pred_yaw_value)) 
		yaw_accuracy = np.minimum(1.0, np.maximum(0.0, ((VP_HOR_SPAN + ET_HOR_SPAN)/2.0 - yaw_distance)/VP_HOR_SPAN))	
		pitch_distance = np.abs(pred_pitch_value - real_pitch_trace[i])
		pitch_accuracy = np.minimum(1.0, np.maximum(0.0, (VP_VER_SPAN + ET_VER_SPAN)/2.0 - pitch_distance)/VP_VER_SPAN)
		area_accuracy = yaw_accuracy * pitch_accuracy
		eff_frame_rate = (rate/VIDEO_FPS)*area_accuracy
		black = 1 - area_accuracy
		# assert eff_frame_rate > 0
		if area_accuracy == 0:
			quality_sum += get_quality(0.0, 0.0, black, True)
		else:
			quality_sum += get_quality(eff_frame_rate, 0.0, black)

	per_frame_quality = quality_sum/VIDEO_FPS
	# print(per_frame_quality)
	return per_frame_quality

##############
# Deprecated
def show_360_buffer(streaming, video_length):
	buffer_info = streaming.buffer_history
	# print(len(buffer_info))
	print(buffer_info)
	if np.ceil(streaming.display_time) < video_length:
		time_to_end = video_length - int(np.ceil(streaming.display_time))
		for i in range(time_to_end):
			buffer_info.append([ np.maximum(buffer_info[-1][0]-1, 0), buffer_info[-1][1]+1])
	# print(buffer_info)
	# print(streaming.display_time)
	print(len(buffer_info))
	global FIGURE_NUM
	a = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM +=1
	plt.plot(range(1, video_length+1), [row[0] for row in buffer_info],'k*-', markersize = 6, markeredgewidth = 0.5, \
			linewidth = 2, markeredgecolor = 'black',  label='BT Buffer Length')
	# plt.plot(range(1, video_length+1), [row[1] for row in buffer_info],'ro-', markersize = 2, markeredgewidth = 0.5, \
	# 		linewidth = 2, markeredgecolor = 'red',  label='ET Buffer Length')

	plt.legend(loc='upper right', fontsize=30)
	# plt.title('BT/ET Buffer Length', fontsize=30)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Second', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, video_length+1, 50))
	plt.yticks(np.arange(0, 21, 5))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085, right=0.97)	
	plt.axis([0, video_length, 0, 30])
	return a 

def show_360_result(streaming, video_length, inti_buffer_length):
	print("length of record is: %s" % len(streaming.evr_recordset))
	deliver_bitrate = [0.0]*video_length
	display_bitrate = [0.0]*video_length
	quality = [0.0]*video_length
	new_quality = [0.0]*video_length
	info = streaming.evr_recordset
	rate = streaming.rates
	rebuff = 0.0
	for i in range(inti_buffer_length):
		deliver_bitrate[i] += rate[-3]
		display_bitrate[i] += VP_BT_RATIO * rate[-3]
		quality[i] += get_quality(VP_BT_RATIO*rate[-3]/VIDEO_FPS, 0.0, 0.0) 
		new_quality[i] += new_get_quality(VP_BT_RATIO*rate[-3], 0.0, 0.0)

	for i in range(len(info)):
		deliver_bitrate[info[i][0]] += rate[info[i][1]]
		display_bitrate[info[i][0]] += VP_BT_RATIO * rate[info[i][1]]
		quality[info[i][0]] += get_quality((VP_BT_RATIO * rate[info[i][1]])/VIDEO_FPS, info[i][2], 0.0) 
		new_quality[i] += new_get_quality(VP_BT_RATIO * rate[info[i][1]], info[i][2], 0.0)
		rebuff += info[i][2]
		if info[i][2] != 0:
			print(i)
		if info[i][2] != 0:
			print(info[i][0])
			print(info[i][2])
			print(len(streaming.buffer_history))

	# figures = []
	# figures.append([show_360_buffer(streaming, video_length), 'buffer'])

	# if IS_SAVING:
	# 	for fig in figures:
	# 		fig[0].savefig('./figures/naive360'+fig[1]+str(streaming.network_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
		
	# if SHOW_FIG:
	# 	show_figure(figures)	

	print("Average Quality (per frame, right): ", sum(quality)/video_length)
	print("Average Quality (new rate): ", sum(new_quality)/video_length)		
	print("Displayed rate sum: ", sum(display_bitrate))
	print("Average Displayed rate: ", sum(display_bitrate)/video_length)	
	print("Received rate sum: ", sum(deliver_bitrate))
	print("Average received rate: ", sum(deliver_bitrate)/video_length)	
	print("Total rebuffering is: %s" % rebuff)


	global FIGURE_NUM
	g = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM += 1
	plt.plot(range(1,video_length+1), display_bitrate, 'r-', linewidth = 2, label='Rendered Video Bitrate')

	plt.plot(range(1,video_length+1), deliver_bitrate, 'b-', linewidth = 2, label='Received Video Bitrate')
	plt.legend(loc='upper right', fontsize=22, ncol=2)
	# plt.title('Effective & Received Video Bitrate')
	plt.xlabel('Second', fontsize =30)
	plt.ylabel('Mbps',fontsize = 30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, video_length+1, 50))
	plt.yticks(np.arange(400, 1201, 400))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.1, right=0.97)	
	plt.axis([0, video_length, 0, 1100])

	g.show()
	raw_input()
	if IS_SAVING_STATIC:
		if not REVISION:
			g.savefig('./figures/naive360/naive_'+str(streaming.network_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
		else:
			g.savefig('./figures/naive360/naive_'+str(streaming.network_file)+'_revision.eps', format='eps', dpi=1000, figsize=(30, 10))

def show_fov_result(streaming, video_length, inti_buffer_length):
	print("length of record is: %s" % len(streaming.evr_recordset))
	deliver_bitrate = [0.0]*video_length
	display_bitrate = [0.0]*video_length
	quality = [0.0]*video_length
	new_quality = [0.0]*video_length
	per_frame_quality = [0.0]*video_length
	black_record = [0.0]*video_length
	freezing_record = [0.0]*video_length
	q_r = [0.0]*video_length
	info = streaming.evr_recordset
	rate = streaming.rates
	black_count = 0
	# rebuff = 0.0
	for i in range(inti_buffer_length):
		deliver_bitrate[i] += rate[-3]
		display_bitrate[i] += VP_ET_RATIO * rate[-3]
		quality[i] += get_quality(VP_ET_RATIO*rate[-3], 0.0, 0.0) 
		new_quality[i] += new_get_quality(VP_ET_RATIO*rate[-3], 0.0, 0.0)
		q_r[i] += get_quality(VP_ET_RATIO*rate[-3], 0.0, 0.0)

	for i in range(len(info)):
		black = 0.0
		index = info[i][0]
		version = info[i][1]
		if version >= 0:
			start_frame_index = int(info[i][0]*VIDEO_FPS)
			end_frame_index = int((1 + info[i][0])*VIDEO_FPS)
			quan_yaw = info[i][3]
			real_yaw = info[i][4]
			real_yaw_trace = streaming.yaw_trace[start_frame_index:end_frame_index]
			quan_pitch = info[i][5]
			real_pitch = info[i][6]
			real_pitch_trace = streaming.pitch_trace[start_frame_index:end_frame_index]
			accuracy = cal_accuracy(real_yaw, quan_yaw, real_pitch, quan_pitch, real_yaw_trace, real_pitch_trace, 1.0)
			assert accuracy <= 1 and accuracy >= 0
			black = 1.0 - accuracy
			black_count += 1
			deliver_bitrate[index] += rate[version]
			display_bitrate[index] += VP_ET_RATIO * accuracy * rate[version]
			if accuracy == 0:
				quality[index] += get_quality(VP_ET_RATIO * accuracy * rate[version], 0.0, black, True)
				new_quality[index] += new_get_quality(0.0, 0.0, black, True) 
			else:
				quality[index] += get_quality(VP_ET_RATIO * accuracy * rate[version], 0.0, black) 
				new_quality[index] += new_get_quality(VP_ET_RATIO * accuracy * rate[version], 0.0, black) 
				per_frame_quality[index] += frame_nlc_fov_quality(quan_yaw, quan_pitch, real_yaw_trace, real_pitch_trace, 1.0, VP_ET_RATIO * rate[version])
				q_r[index] += new_get_quality(VP_ET_RATIO * accuracy * rate[version], 0.0, 0.0)
		else:
			deliver_bitrate[index] += 0.0
			display_bitrate[index] += 0.0
			quality[index] += get_quality(0.0, 1.0, black, True)
			per_frame_quality[index] += get_quality(0.0, 1.0, black, True)
			new_quality[index] += new_get_quality(0.0, 1.0, black, True) 
			freezing_record[index] += 1
		black_record[index] += black

	print("Average Quality on chunk (wrong): ", sum(quality)/video_length)	
	print("Average Quality (per frame, right): ", sum(per_frame_quality)/video_length)	
	print("Average Quality (new rate): ", sum(new_quality)/video_length)
	print("Average Q_r for FOV only", sum(q_r)/video_length)		
	print("Displayed rate sum: ", sum(display_bitrate))
	print("Average Displayed rate: ", sum(display_bitrate)/video_length)	
	print("Received rate sum: ", sum(deliver_bitrate))
	print("Average received rate: ", sum(deliver_bitrate)/video_length)	
	# print(black_record)
	print("Total freezing time is : %s, %s%%" % (np.sum(freezing_record), 100.0*np.sum(freezing_record)/video_length))
	print("Average black ratio is: %s" % (np.sum(black_record)/black_count))

	global FIGURE_NUM
	g = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM += 1
	plt.plot(range(1,video_length+1), display_bitrate, 'r-', linewidth = 2, label='Rendered Video Bitrate')

	plt.plot(range(1,video_length+1), deliver_bitrate, 'b-', linewidth = 2, label='Received Video Bitrate')
	plt.legend(loc='upper right', fontsize=22, ncol=2)
	# plt.title('Effective & Received Video Bitrate')
	plt.xlabel('Second', fontsize =30)
	plt.ylabel('Mbps',fontsize = 30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, video_length+1, 50))
	plt.yticks(np.arange(400, 1201, 400))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.1, right=0.97)	
	plt.axis([0, video_length, 0, 1000])

	g.show()
	raw_input()
	if IS_SAVING_STATIC:
		if not REVISION:
			g.savefig('./figures/predictive/pred_'+str(streaming.network_file)+'_'+str(streaming.fov_file)+'.eps', format='eps', dpi=1000, figsize=(30, 10))
		else:
			g.savefig('./figures/predictive/pred_'+str(streaming.network_file)+'_'+str(streaming.fov_file)+'_revision.eps', format='eps', dpi=1000, figsize=(30, 10))
