import numpy as np
import math

VIDEO_LEN = 300


BITRATE_LEN = 4
INIT_BL_RATIO = 0.1
INIT_EL_RATIO = 0.9
EL_LOWEST_RATIO = 0.75
EL_HIGHER_RATIO = 1.25

# BW prediction
BW_PRED_SAMPLE_SIZE = 10	# second
INIT_BW = 30

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
	if vp_trace == 'Video_9_alpha_beta_new.mat':
		yaw_trace_data = [x + 90 for x in yaw_trace_data]
		
		for i in range(len(yaw_trace_data)):
			if yaw_trace_data[i] >= 180:
				yaw_trace_data[i] -= 360
			assert yaw_trace_data[i] >= -180:
	return yaw_trace_data[:VIDEO_LEN*VIDEO_FPS], pitch_trace_data[:VIDEO_LEN*VIDEO_FPS]


def load_init_rates(average_bw):
	rate_cut = [0.0] * BITRATE_LEN
	rate_cut[0] = INIT_BL_RATIO * average_bw
	rate_cut[1] = INIT_EL_RATIO * EL_LOWEST_RATIO * average_bw
	rate_cut[2] = INIT_EL_RATIO * average_bw
	rate_cut[3] = INIT_EL_RATIO * EL_HIGHER_RATIO * average_bw
	return rate_cut

def generate_video_trace(rate_cut):
	video_trace0 = rate_cut[0] * np.ones(VIDEO_LEN)
	video_trace1 = rate_cut[1] * np.ones(VIDEO_LEN)
	video_trace2 = rate_cut[2] * np.ones(VIDEO_LEN)
	video_trace3 = rate_cut[3] * np.ones(VIDEO_LEN)
	return [video_trace0, video_trace1, video_trace2, video_trace3]


def predict_bw(bw_history):
	if len(bw_history) == 0:
		return INIT_BW
	else:
		if len(bw_history) < BW_PRED_SAMPLE_SIZE:
			return sum(row[0] for row in bw_history)/len(bw_history)
		else:
			return sum(row[0] for row in bw_history[-BW_PRED_SAMPLE_SIZE:])/BW_PRED_SAMPLE_SIZE

