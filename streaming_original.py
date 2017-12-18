
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Display buffer initialization
BUFFER_BL_INIT = 20
BUFFER_EL_INIT = 1
Q_REF_BL = 5
Q_REF_EL = 1
ET_MAX_PRED = Q_REF_EL + 1

# Control parameters
IS_DEBUGGING = 1
DISPLAYRESULT = 1
TRACE_IDX = 1
KP = 0.8		# P controller
KI = 0.01		# I controller

# Video parameters
VIDEO_LEN = 600		# seconds
VIDEO_FPS = 30		# hz	
NETWORK_TRACE_LEN = VIDEO_LEN 		# seconds
VIEW_PRED_SAMPLE_LEN = 30	# samples used for prediction
POLY_ORDER = 1				#  1: linear   2: quadratic
FRAME_MV_LIMIT = 180		# horizontal motion upper bound in degree, used for horizontal circular rotation
KB_IN_MB = 1000

#
INIT_BW = 100
BW_PRED = 2
BW_PRED_SAMPLE_SIZE = 1

# Directory
NETWORK_TRACE_FILENAME = 'BW_Trace_8.mat'
VIEWPORT_TRACE_FILENAME1 = 'view_angle_combo_video1.mat'
VIEWPORT_TRACE_FILENAME2 = 'view_trace_fanyi_amsterdam_2D.mat'


class streaming(object):

	def __init__(self):
		self.video_seg_index_BL = BUFFER_BL_INIT 
		self.video_seg_index_EL = BUFFER_EL_INIT
		self.network_seg_index = 1
		self.remaining_time = 1
		self.video_download_timestamp = 0
		self.buffer_size_BL = BUFFER_BL_INIT
		self.buffer_size_EL = BUFFER_EL_INIT
		self.buffer_size_history = []
		self.downloadedPartialVideo = 0

		self.EVR_BL_Recordset = []
		self.EVR_EL_Recordset = []
		self.video_segment = 0
		self.GBUFFER_BL_EMPTY_COUNT = 0
		self.GBUFFER_EL_EMPTY_COUNT = 0
		self.video_version = 0
		self.video_seg_index = 0
		self.yaw_predict_value = 0
		self.pitch_predict_value = 0
		self.yaw_predict_value_quan = 0
		self.pitch_predict_value_quan = 0
		self.record_info = []
		self.video_bw_history = []

	def run(self, network_trace, yaw_trace, pitch_trace, video_trace, rate_cut):
		while self.video_seg_index_BL < VIDEO_LEN or \
			(self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
			if not self.downloadedPartialVideo:

				if BW_PRED == 1:
					sniff_BW = self.getCurrentBW(network_trace)
				elif BW_PRED == 2:
					sniff_BW = self.predictBW()

				self.control(rate_cut, sniff_BW)
				self.video_segment = video_trace[self.video_version][self.video_segment_index]
				if self.video_version >= 1:
					self.predict_yaw(yaw_trace)
					self.predict_pitch(pitch_trace)
					
			self.video_fetching(network_trace, rate_cut)
			# return
			if not self.downloadedPartialVideo:
				self.record_info = np.append(self.record_info, \
					[self.video_download_timestamp, \
					self.yaw_predict_value, self.pitch_predict_value, \
					yaw_trace[int(self.video_download_timestamp*VIDEO_FPS)], pitch_trace[int(self.video_download_timestamp*VIDEO_FPS)], \
					network_trace[int(self.video_download_timestamp)], sniff_BW, \
					self.buffer_size_BL, self.buffer_size_EL])

		print(self.network_seg_index)
		print('Simluation done')
		# print(len(self.EVR_BL_Recordset))	

	def control(self, rate_cut, sniff_BW):
		# print(self.buffer_size_BL)
		# print(self.video_seg_index_BL)
		# print(self.downloadedPartialVideo)
		if not self.downloadedPartialVideo:
			if self.buffer_size_BL < Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN :
				self.video_version = 0
				self.video_segment_index = self.video_seg_index_BL
				# print(self.video_seg_index_BL, self.video_seg_index_EL, int(np.floor(self.video_download_timestamp)))
			elif (self.buffer_size_BL >= Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN ) \
				or (self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
				#PI control logic
				u_p = KP * (self.buffer_size_EL - Q_REF_EL)
				u_i = 0
				if len(self.buffer_size_history) != 0:
					# print(self.buffer_size_history)
					for index in range(0, len(self.buffer_size_history)):
						u_i +=  KI  * (self.buffer_size_history[index][1] - Q_REF_EL)
				
				u = u_i + u_p

				########################
				# if self.buffer_size_EL >= 1:
				# 	v = u + 1
				# 	delta_time = 1
				# else :
				# 	v = u + np.ceil(self.video_download_timestamp + (10.**-8)) - self.video_download_timestamp
				# 	delta_time = np.ceil(self.video_download_timestamp) - self.video_download_timestamp
				# R_hat = v * sniff_BW
				###############
				v = u + 1
				delta_time = self.buffer_size_EL
				R_hat = min(v, delta_time) * sniff_BW
				#########
				# print(R_hat, sniff_BW, self.video_seg_index_EL)
				if R_hat >= rate_cut[3] - rate_cut[0]:
					current_video_version = 3
				elif R_hat >= rate_cut[2] - rate_cut[0]:
					current_video_version = 2
				else:
					current_video_version = 1

				# if len(self.video_bw_history) != 0:
				# 	self.video_version = min(current_video_version, self.video_bw_history[-1][2] + 1)
				# else: 
				self.video_version = current_video_version
				self.video_segment_index = self.video_seg_index_EL
				# print('time:', self.video_download_timestamp)
		else:
			print("Still download previous segment")
		return 

	def video_fetching(self, network_trace, rate_cut):
		temp_video_download_timestamp = self.video_download_timestamp
		video_rate = 0  #### Liyang
		# print(self.video_segment, network_trace[self.network_seg_index]*self.remaining_time)
		if network_trace[self.network_seg_index]*self.remaining_time >= self.video_segment:
			if self.video_version == 0:
				self.EVR_BL_Recordset.append([self.video_seg_index_BL, self.video_version, network_trace[self.network_seg_index]])
				self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
				self.remaining_time -= (self.video_download_timestamp - temp_video_download_timestamp)
				# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
				self.downloadedPartialVideo = 0
				self.video_seg_index_BL += 1
				self.buffer_size_BL += 1
				video_rate = rate_cut[0] ### Liyang
			elif self.video_version >= 1:
				if self.buffer_size_EL <=  ET_MAX_PRED:
					self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index]])
					self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
					self.remaining_time -= (self.video_download_timestamp - temp_video_download_timestamp)
					# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
					self.downloadedPartialVideo = 0
					self.video_seg_index_EL += 1
					self.buffer_size_EL += 1
				else:
					self.remaining_time = 1
					self.downloadedPartialVideo = 0
					self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
					self.video_segment = 0
					self.flag = 1
				
				#### Liyang
				video_rate = rate_cut[self.video_version] - rate_cut[0]
				####

			else :
				print("Unknown video version.")
			
			if not self.downloadedPartialVideo:
				if self.video_segment > 0 :
					if len(self.video_bw_history) == 0:
						bw = video_rate/self.video_download_timestamp 
					else :
						bw = video_rate/(self.video_download_timestamp - self.video_bw_history[-1][1])
					
					self.video_bw_history.append([bw, self.video_download_timestamp, self.video_version])
					# print(bw,network_trace[self.network_seg_index], int(np.floor(temp_video_download_timestamp)))
				else:
					self.video_bw_history.append([network_trace[self.network_seg_index], self.video_download_timestamp, 100])

		else:
			# if self.video_version == 0:
			# 	print('Download base tier, bandwidth is not enough.')
			# elif self.video_version >= 1:
			# 	print('Download enhancement tier, bandwidth is not enough.')
			self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
			self.video_segment = self.video_segment - network_trace[self.network_seg_index]*self.remaining_time
			self.remaining_time = 1
			self.downloadedPartialVideo = 1
			# self.network_seg_index += 1

		##########################
		# if np.floor(self.video_download_timestamp) != np.floor(temp_video_download_timestamp):
		# 	if self.buffer_size_BL != 0:
		# 		self.buffer_size_BL -= 1
		# 	else:
		# 		self.GBUFFER_BL_EMPTY_COUNT += 1
		# 		self.video_seg_index_BL += 1
			
		# 	if self.buffer_size_EL != 0:
		# 		self.buffer_size_EL -= 1
		# 	else:
		# 		self.GBUFFER_EL_EMPTY_COUNT += 1
		# 		self.video_seg_index_EL += 1
		# 	self.buffer_size_history.append([self.buffer_size_BL, self.buffer_size_EL])
		# 	self.network_seg_index += 1

		##########################

		self.buffer_size_BL = max(0, self.buffer_size_BL - (self.video_download_timestamp - temp_video_download_timestamp))
		self.buffer_size_EL = max(0, self.buffer_size_EL - (self.video_download_timestamp - temp_video_download_timestamp))
		
		if np.floor(self.video_download_timestamp) != np.floor(temp_video_download_timestamp):
			if self.buffer_size_BL == 0:
				self.GBUFFER_BL_EMPTY_COUNT += 1
				self.video_seg_index_BL = int(np.floor(self.video_download_timestamp))
			if self.buffer_size_EL == 0:
				self.GBUFFER_EL_EMPTY_COUNT += 1
				self.video_seg_index_EL = int(np.floor(self.video_download_timestamp))
			self.buffer_size_history.append([self.buffer_size_BL, self.buffer_size_EL])
			self.network_seg_index += 1

		return 

	def predict_yaw(self, yaw_trace):
		# print(yaw_trace)
		if self.video_download_timestamp < 1:
			self.yaw_predict_value = 360 
		else:
			if not self.downloadedPartialVideo:
				vp_index = np.arange(-VIEW_PRED_SAMPLE_LEN,0) + int(self.video_download_timestamp*VIDEO_FPS)
				# print(vp_index)
				vp_value = []
				for index in vp_index:
					vp_value.append(yaw_trace[index])
				# print(vp_value)
				for value in vp_value[1:]:
					if value - vp_value[vp_value.index(value)-1] > FRAME_MV_LIMIT:
						value -= 360
					elif vp_value[vp_value.index(value)-1] - value > FRAME_MV_LIMIT:
						value += 360
				# print(vp_index, vp_value)
				yaw_predict_model = np.polyfit(vp_index, vp_value, POLY_ORDER)
				yaw_predict_idx = self.video_seg_index_EL*VIDEO_FPS - VIDEO_FPS/2
				self.yaw_predict_value = np.round(np.polyval(yaw_predict_model,yaw_predict_idx))
				# print(yaw_predict_value)
				# Adjust yaw_predict value to range from 1 to 360
			else: return
		self.yaw_predict_value %= 360
		if self.yaw_predict_value == 0: self.yaw_predict_value += 360
		
		# quantize yaw predict value to range from 1 to 12
		# different with the value in Fanyi's Matlab source code
		self.yaw_predict_value_quan = self.yaw_predict_value / 30
		if self.yaw_predict_value_quan == 12: self.yaw_predict_value_quan = 0
		self.yaw_predict_value_quan += 1
		# print(yaw_predict_value_quan)
		return

	def predict_pitch(self, pitch_trace):
		if self.video_download_timestamp < 1:
			self.pitch_predict_value = 90
		else:
			if not self.downloadedPartialVideo:
				# print(self.video_download_timestamp)
				vp_index = np.arange(-VIEW_PRED_SAMPLE_LEN,0) + int(np.floor(self.video_download_timestamp*VIDEO_FPS))
				vp_value = []
				for index in vp_index:
					vp_value.append(pitch_trace[index])
				pitch_predict_model = np.polyfit(vp_index, vp_value, POLY_ORDER)
				pitch_predict_idx = self.video_seg_index_EL*VIDEO_FPS - VIDEO_FPS/2
				self.pitch_predict_value = np.round(np.polyval(pitch_predict_model, pitch_predict_idx))
			else: return		
		if self.pitch_predict_value in range(0, 46):
			self.pitch_predict_value_quan = 1
		elif self.pitch_predict_value in range(46, 136):
			self.pitch_predict_value_quan = 2
		else:
			self.pitch_predict_value_quan = 3
		return 

	def getCurrentBW(self, network_trace):
		if self.network_seg_index == 0 or np.floor(self.video_download_timestamp) == 0:
			return network_trace[0]
		else:
			# print(int(self.video_download_timestamp))
			# print(network_trace)
			return network_trace[int(np.floor(self.video_download_timestamp))]

	def predictBW(self):
		if len(self.video_bw_history) == 0:
			return INIT_BW
		else:
			# print(int(self.video_download_timestamp))
			# print(network_trace)
			if len(self.video_bw_history) < BW_PRED_SAMPLE_SIZE:
				return sum(row[0] for row in self.video_bw_history)/len(self.video_bw_history)
			else :
				return sum(row[0] for row in self.video_bw_history[-BW_PRED_SAMPLE_SIZE:])/BW_PRED_SAMPLE_SIZE


def main():
	network_trace = loadNetworkTrace()
	yaw_trace, pitch_trace = loadViewportTrace()
	network_pdf, pdf_bins, pdf_patches = plot_pdf(network_trace)
	network_cdf, cdf_bins, cdf_patches = plot_cdf(network_trace)
	rate_cut = rate_determine(network_cdf)
	video_trace = loadVideoTrace(rate_cut)

	streaming_sim = streaming()
	streaming_sim.run(network_trace, yaw_trace, pitch_trace, video_trace, rate_cut)
	# print(len(streaming_sim.buffer_size_history))
	if IS_DEBUGGING:
		display(streaming_sim.record_info, streaming_sim.EVR_BL_Recordset, streaming_sim.EVR_EL_Recordset, \
			rate_cut)
		raw_input()

def loadNetworkTrace():
	mat_contents = sio.loadmat(NETWORK_TRACE_FILENAME)
	trace_data = KB_IN_MB*mat_contents['delta_bit_debugging_buffer'] # array of structures
	# print(type(trace_data), len(trace_data))
	assert (len(trace_data) > NETWORK_TRACE_LEN)
	result = []
	for x in range(0, NETWORK_TRACE_LEN):
		result.append(trace_data[x][0])
	# print(result)
	return result

def loadViewportTrace():
	if TRACE_IDX == 1: 
		mat_contents = sio.loadmat(VIEWPORT_TRACE_FILENAME1)
	else:
		mat_contents = sio.loadmat(VIEWPORT_TRACE_FILENAME2)
	yaw_trace_data = mat_contents['view_angle_yaw_combo'] + 180 # array of structures
	pitch_trace_data = mat_contents['view_angle_pitch_combo'] + 90 # array of structures
	# print(len(yaw_trace_data[0]), len(pitch_trace_data[0]))
	assert (len(yaw_trace_data[0]) > VIDEO_LEN*VIDEO_FPS and len(pitch_trace_data[0]) > VIDEO_LEN*VIDEO_FPS)
	return yaw_trace_data[0][:VIDEO_LEN*VIDEO_FPS], pitch_trace_data[0][:VIDEO_LEN*VIDEO_FPS]

def loadVideoTrace(rate_cut):
	video_trace0 = rate_cut[0] * np.ones(VIDEO_LEN)
	video_trace1 = (rate_cut[1] - rate_cut[0]) * np.ones(VIDEO_LEN)
	video_trace2 = (rate_cut[2] - rate_cut[0]) * np.ones(VIDEO_LEN)
	video_trace3 = (rate_cut[3] - rate_cut[0]) * np.ones(VIDEO_LEN)
	return [video_trace0, video_trace1, video_trace2, video_trace3]

def plot_pdf(trace):
	f = plt.figure(1)
	n, bins, patches = plt.hist(trace, range(0, int(np.ceil(max(trace))) + 1), normed = 1, label='PDF', color='b')
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def plot_cdf(trace):
	f = plt.figure(2)
	n, bins, patches = plt.hist(trace, range(0,int(np.ceil(max(trace))) + 1), normed = 1, cumulative=True, label='CDF', histtype='stepfilled', color='b')
	# print(a,b,c)
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def rate_determine(network_cdf):
	cut_percent = [0.4, 0.6, 0.8]
	rate_cut = [100]
	cut_idx = 0
	for cum in network_cdf:
		if cum >= np.max(network_cdf) * cut_percent[cut_idx]:
			rate_cut.append(np.round(network_cdf.tolist().index(cum)))
			cut_idx += 1
			if cut_idx >= len(cut_percent): 
				break
	if IS_DEBUGGING: 
		print('Rate0 = %d, Rate1 = %d, Rate2 = %d, Rate3 = %d' % (rate_cut[0],rate_cut[1],rate_cut[2],rate_cut[3]))
	return rate_cut

def display(record_info, EVR_BL_Recordset, EVR_EL_Recordset, rate_cut):
	# print(len(record_info)/9)
	# print(len(EVR_EL_Recordset))
	# print(len(EVR_BL_Recordset))

	display_result = record_info.reshape(len(record_info)/9, 9).T
	f = plt.figure(3)
	plt.plot(display_result[0], display_result[1],'r-', display_result[0], display_result[2],'b-', display_result[0], display_result[3],'g-',display_result[0], display_result[4],'k-',)
	g = plt.figure(4)
	plt.plot(display_result[0], display_result[5],'r-', display_result[0], display_result[6],'b-')
	h = plt.figure(5)
	plt.plot(display_result[0], display_result[7],'r-', display_result[0], display_result[8],'b-')
	

	display_bitrate = [0]*VIDEO_LEN
	# print(EVR_BL_Recordset)
	for i in range (0,BUFFER_BL_INIT):
		display_bitrate[i] += rate_cut[0]
	display_bitrate[0] += rate_cut[3]
	for i in range(0, len(EVR_BL_Recordset)):
		display_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]
	for i in range(0,len(EVR_EL_Recordset)):
		display_bitrate[EVR_EL_Recordset[i][0]] += rate_cut[EVR_EL_Recordset[i][1]]
	
	i = plt.figure(6)
	plt.plot(range(1,VIDEO_LEN+1), display_bitrate, 'r-')
	# print(len(display_bitrate))
	# print(EVR_BL_Recordset)
	# print(EVR_EL_Recordset)
	# f.show()
	# g.show()	
	# h.show()
	i.show()
	return

if __name__ == '__main__':
	main()