import numpy as np
import math

VIDEO_LEN = 300
CHUNK_DURATION = 1.0

BITRATE_LEN = 4
INIT_BL_RATIO = 0.1
INIT_EL_RATIO = 0.9
EL_LOWEST_RATIO = 0.75
EL_HIGHER_RATIO = 1.25

# BW prediction
BW_PRED_SAMPLE_SIZE = 10	# second
INIT_BW = 30

# FOV prediction
VIDEO_FPS = 30
VIEW_PRED_SAMPLE_LEN = 30
FRAME_MV_LIMIT = 180.0
POLY_ORDER = 1

def show_network(bandwidth_trace, delay_trace):
	print("5G trace mean:", np.mean(network_trace))
	print("5G trace standard deviation:", np.std(network_trace))
	print("5G trace peak:", np.max(network_trace))
	print("5G trace min:", np.min(network_trace))
	print("5G trace median:", np.median(network_trace))

	print("5G delay mean:", np.mean(network_delay))
	print("5G delay standard deviation:", np.std(network_delay))
	print("5G delay peak:", np.max(network_delay))
	print("5G delay min:", np.min(network_delay))
	print("5G delay median:", np.median(network_delay))
	return np.mean(network_trace)




def load_init_rates(average_bw):
	rate_cut = [0.0] * BITRATE_LEN
	rate_cut[0] = INIT_BL_RATIO * average_bw
	rate_cut[1] = INIT_EL_RATIO * EL_LOWEST_RATIO * average_bw
	rate_cut[2] = INIT_EL_RATIO * average_bw
	rate_cut[3] = INIT_EL_RATIO * EL_HIGHER_RATIO * average_bw
	return rate_cut

def generate_video_trace(rate_cut):
	video_trace0 = rate_cut[0] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	video_trace1 = rate_cut[1] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	video_trace2 = rate_cut[2] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	video_trace3 = rate_cut[3] * CHUNK_DURATION * np.ones(VIDEO_LEN)
	return [video_trace0, video_trace1, video_trace2, video_trace3]

def load_viewport(vp_trace):
	mat_contents = sio.loadmat(vp_trace)
	trace_data = mat_contents['data_alpha_beta'] # array of structures
	# pitch_trace_data = mat_contents['view_angle_pitch_combo'] + 90 # array of structures
	# print(len(yaw_trace_data[0]), len(pitch_trace_data[0]))
	print(trace_data.T.shape)

	yaw_trace_data = (trace_data.T[1]/math.pi)*180+180
	pitch_trace_data = (trace_data.T[2]/math.pi)*180
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
	if pitch_predict_value > 90:
		pitch_predict_quan = 2
	else:
		pitch_predict_quan = 1

	return pitch_predict_value, pitch_predict_quan

def predict_bw(bw_history):
	if len(bw_history) == 0:
		return INIT_BW
	else:
		if len(bw_history) < BW_PRED_SAMPLE_SIZE:
			return sum(row[0] for row in bw_history)/len(bw_history)
		else:
			return sum(row[0] for row in bw_history[-BW_PRED_SAMPLE_SIZE:])/BW_PRED_SAMPLE_SIZE

