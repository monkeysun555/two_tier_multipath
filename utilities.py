# Utilities for 5G dynamic related simulations
import numpy as np
import math
import scipy.io as sio
import matplotlib.pyplot as plt

# Video info
VIDEO_LEN = 300
CHUNK_DURATION = 1.0

# For alpha/gamma
# ALPHA_CAL_LEN is the length of calculating alpha, in seconds
# GAMMA_CAL_LEN is the length of calculating gamma, in seconds
ALPHA_DYNAMIC = 1	# <======================= alpha control
BUFFER_RANGE = 4
ALPHA_CAL_LEN = 10
GAMMA_CAL_LEN = 10
BW_THRESHOLD = 0.2

# Rate Allocation
BITRATE_LEN = 4
INIT_BL_RATIO = 0.1
INIT_EL_RATIO = 0.9
EL_LOWEST_RATIO = 0.75
EL_HIGHER_RATIO = 1.25
BW_UTI_RATIO = 0.85
BW_DALAY_RATIO = 0.95

# BW prediction
BW_PRED_SAMPLE_SIZE = 10	# second
INIT_BW = 30

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

# Streaming 
BUFFER_BL_INIT = 10
BUFFER_EL_INIT = 1
R_BASE = 100.0

# Plot info
FIGURE_NUM = 1
SHOW_FIG = 1


def rate_optimize(display_time, average_bw, var_bw, alpha_history, version):
	gamma_curve = update_gamma(average_bw, var_bw)
	alpha_curve = update_alpha(display_time, alpha_history, version)
	alpha_gamma = np.multiply(alpha_curve, gamma_curve)
	optimal_alpha_gamma = np.amax(alpha_gamma)
	optimal_buffer_len = np.argmax(alpha_gamma)
	beta = ((1 - optimal_alpha_gamma)*(ET_VER_SPAN*ET_HOR_SPAN))/ \
			(optimal_alpha_gamma*(BT_VER_SPAN*BT_HOR_SPAN - ET_HOR_SPAN*ET_VER_SPAN))

	rate_et_average = BW_UTI_RATIO*average_bw/ \
					(1 + (BT_HOR_SPAN*BT_VER_SPAN*beta)/((1-beta)*ET_HOR_SPAN*ET_VER_SPAN))
	rate_bt_average = BW_UTI_RATIO*average_bw - rate_et_average
	rate_et_low = rate_et_average * EL_LOWEST_RATIO
	rate_et_high = rate_et_average * EL_HIGHER_RATIO
	#update rate cut and rate cut version
	print("Do a optimzation, current time is %s" % display_time)
	print("Alpha is: ", alpha_curve)
	print("Gamma is: ", gamma_curve)
	print("alpha gamma is: ", alpha_gamma)
	print("average bw is %s, rate_bt is: %s and rate_et is: %s" %(average_bw, rate_bt_average, rate_et_average))
	print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	print("<++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++>")
	return [rate_bt_average, rate_et_low, rate_et_average, rate_et_high], optimal_buffer_len


def cal_average_bw(network_time, bw_history, last_ref_time):
	average_gamma_bw = 0.0
	variance_gamma_bw = 0.0
	average_gamma_real = 0.0
	if network_time - GAMMA_CAL_LEN < last_ref_time:
		average_gamma_bw = -1.0
		variance_gamma_bw = -1.0
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
		variance_gamma_bw = np.var(temp_bw)/average_gamma_bw
		average_gamma_real = np.sum(temp_real)/len(temp_real)
	return average_gamma_bw, variance_gamma_bw, average_gamma_real

def is_bw_change(current_bw, ref_bw):
	if current_bw >= (1+BW_THRESHOLD)*ref_bw or current_bw <= (1-BW_THRESHOLD)*ref_bw:
		return True
	else:
		return False

def record_alpha(yaw_trace, pitch_trace, display_time, buffer_range = BUFFER_RANGE):
	temp_alpha_len = np.minimum(buffer_range, VIDEO_LEN - int(round(display_time)))
	temp_alpha = [[0] * 5 for i in range(temp_alpha_len)]
	for i in range(temp_alpha_len):
		temp_yaw_value, temp_yaw_quan = predict_yaw_trun(yaw_trace, display_time-0.5, int(round(display_time)) + i)
		temp_pitch_value, temp_pitch_quan = predict_pitch_trun(pitch_trace, display_time-0.5, int(round(display_time)) + i)
		temp_alpha[i][0] = temp_yaw_value
		temp_alpha[i][1] = temp_yaw_quan
		temp_alpha[i][2] = temp_pitch_value
		temp_alpha[i][3] = temp_pitch_quan
	return temp_alpha


def update_gamma(average_bw, var_bw):
	# Fix gamma
	return [0.747, 0.846, 0.903, 0.938]

def update_alpha(display_time, alpha_history, version, buffer_range = BUFFER_RANGE):
	alpha = [0.0]*buffer_range
	if ALPHA_DYNAMIC:
		alpha_calculation_len = 0
		for i in reversed(range(len(alpha_history)-1)):
			if alpha_history[i][2] != version:
				break
			alpha_calculation_len += 1	

		alpha_calculation_len = np.minimum(alpha_calculation_len, ALPHA_CAL_LEN)
		# print("alpha_history is: ", alpha_history, alpha_calculation_len)
		assert alpha_calculation_len >= 1
		if version > 0:
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
			for i in range(buffer_range):
				temp_alpha = 0.0
				temp_alpha_count = 0
				for j in range(alpha_calculation_len-i):
					temp_alpha += alpha_history[-(j+i+2)][1][i][4]
					temp_alpha_count += 1
				if temp_alpha_count == 0:
					alpha[i] = 0.0
				else:
					alpha[i] = temp_alpha/temp_alpha_count
	else:
		alpha = [0.942, 0.896, 0.865, 0.825]
	
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
	return average_bw, np.var(bw_record)/average_bw, np.sum(bw_real)/len(bw_real) 


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
	return np.mean(network_trace)

def load_init_rates(average_bw):
	rate_cut = [0.0] * BITRATE_LEN
	rate_cut[0] = INIT_BL_RATIO * BW_UTI_RATIO * average_bw
	rate_cut[1] = INIT_EL_RATIO * EL_LOWEST_RATIO * BW_UTI_RATIO * average_bw
	rate_cut[2] = INIT_EL_RATIO * BW_UTI_RATIO * average_bw
	rate_cut[3] = INIT_EL_RATIO * EL_HIGHER_RATIO * BW_UTI_RATIO * average_bw
	return rate_cut

def generate_video_trace(rate_cut):
	video_trace0 = rate_cut[0] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	video_trace1 = rate_cut[1] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	video_trace2 = rate_cut[2] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	video_trace3 = rate_cut[3] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	return [video_trace0, video_trace1, video_trace2, video_trace3]

def predict_bw(bw_history):
	if len(bw_history) == 0:
		return INIT_BW
	else:
		if len(bw_history) < BW_PRED_SAMPLE_SIZE:
			return sum(row[0] for row in bw_history)/len(bw_history)
		else:
			return sum(row[0] for row in bw_history[-BW_PRED_SAMPLE_SIZE:])/BW_PRED_SAMPLE_SIZE

def load_viewport(vp_trace):
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
	return yaw_trace_data[:VIDEO_LEN*VIDEO_FPS], pitch_trace_data[:VIDEO_LEN*VIDEO_FPS]

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

def show_rates(streaming):
	# For video rate
	receive_bitrate = [0.0]*VIDEO_LEN
	display_bitrate = [0.0]*VIDEO_LEN
	log_bitrate = [0.0]*VIDEO_LEN
	total_alpha = 0.0	# fov accuracy ratio
	total_gamma = 0.0	# chunk pass ratio
	bl_info = streaming.evr_bl_recordset
	el_info = streaming.evr_el_recordset

	# <update> later
	rate_cut = streaming.rate_cut

	for i in range(BUFFER_BL_INIT):
		display_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]
		receive_bitrate[i] += VP_BT_RATIO * rate_cut[0][0]

	for i in range(BUFFER_EL_INIT):
		display_bitrate[i] += VP_ET_RATIO * rate_cut[0][-1]
		receive_bitrate[i] += VP_ET_RATIO * rate_cut[0][-1]

	for i in range(len(bl_info)):
		display_bitrate[bl_info[i][0]] += VP_BT_RATIO * rate_cut[bl_info[i][5]][bl_info[i][1]]
		receive_bitrate[bl_info[i][0]] += VP_BT_RATIO * rate_cut[bl_info[i][5]][bl_info[i][1]]

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
		total_alpha += el_accuracy
		total_gamma += time_eff


	for i in range(len(display_bitrate)):
		if display_bitrate[i] == 0.0:
			print("cannot be like this")
			continue
		log_bitrate[i] += math.log10(display_bitrate[i]/R_BASE)

	print("Average alpha ratio: ", (total_alpha + 1.0)/VIDEO_LEN)
	print("Average effective alpha: ", (total_alpha + 1.0)/(len(el_info) + 1))
	print("EL existing ratio: ", (len(el_info)+1.0)/VIDEO_LEN)
	print("Average gamma: ", (total_gamma + 1.0)/VIDEO_LEN)

	print("Displayed effective rate sum: ", sum(display_bitrate))
	print("Received effective rate sum: ", sum(receive_bitrate))
	print("Log rate sum: ", sum(log_bitrate))

	global FIGURE_NUM
	g = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM += 1
	plt.plot(range(1,VIDEO_LEN+1), display_bitrate, 'bo-', markersize = 3,\
	 markeredgewidth = 0.5, markeredgecolor = 'blue', linewidth = 2, label='Displayed Effective Video Bitrate')
	plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r*-', markersize = 8, \
	 markeredgewidth = 0.5, markeredgecolor = 'red', linewidth = 2, label='Received Effective Video Bitrate')
	plt.legend(loc='upper right', fontsize=30)
	# plt.title('Effective & Received Video Bitrate')
	plt.xlabel('Second', fontsize =30)
	plt.ylabel('Mbps',fontsize = 30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(0, 1201, 400))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085, right=0.97)	
	plt.axis([0, 300, 0, max(receive_bitrate) + 600])

	return g

def show_buffer(streaming):
	## Plot buffer length
	buffer_info = streaming.buffer_history
	if np.ceil(streaming.display_time) < VIDEO_LEN:
		time_to_end = VIDEO_LEN - int(np.ceil(streaming.display_time))
		for i in range(time_to_end):
			buffer_info.append([ np.maximum(buffer_info[-1][0]-1, 0),  np.maximum(buffer_info[-1][1]-1, 0), buffer_info[-1][2]+i+1])
	# print(buffer_info)
	# print(streaming.display_time)
	# print(len(buffer_info))
	global FIGURE_NUM
	a = plt.figure(FIGURE_NUM,figsize=(20,5))
	FIGURE_NUM +=1
	plt.plot(range(1, VIDEO_LEN+1), [row[0] for row in buffer_info],'r*-', markersize = 6, markeredgewidth = 0.5, \
			linewidth = 2, markeredgecolor = 'red',  label='BT Buffer Length')
	plt.plot(range(1, VIDEO_LEN+1), [row[1] for row in buffer_info],'bo-', markersize = 2, markeredgewidth = 0.5, \
			linewidth = 2, markeredgecolor = 'blue',  label='ET Buffer Length')

	plt.legend(loc='upper right', fontsize=30)
	# plt.title('BT/ET Buffer Length', fontsize=30)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Second', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(0, 21, 5))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085, right=0.97)	
	plt.axis([0, 300, 0, 20])

	return a 

def show_bw(streaming):
	for i in range(len(streaming.video_bw_history)):
		print("Bw at %s is %s and real is %s.\n" % (streaming.video_bw_history[i][1], \
									streaming.video_bw_history[i][0], streaming.video_bw_history[i][4]))
	

def show_figure(figures):
	for f in figures:
		f.show()
	raw_input()

def show_result(streaming):
	# record figures
	figures = []
	figures.append(show_buffer(streaming))
	#	display and received bitrates
	figures.append(show_rates(streaming))

	# show_bw(streaming)

	# for i in streaming.alpha_history:
	# 	print(i)
	
	if SHOW_FIG:
		show_figure(figures)


	return

