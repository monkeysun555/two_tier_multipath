
# coding: utf-8

# In[1]:


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

# Display buffer initialization
BUFFER_BL_INIT = 20
BUFFER_EL_INIT = 1
Q_REF_BL = 10
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
VIEW_PRED_SAMPLE_LEN = 30  # samples used for prediction
POLY_ORDER = 1  # 1: linear   2: quadratic
# horizontal motion upper bound in degree, used for horizontal circular rotation
FRAME_MV_LIMIT = 180
KB_IN_MB = 100   			# actually not

#
INIT_BW = 100
BW_PRED = 2
BW_PRED_SAMPLE_SIZE = 5

# Directory
NETWORK_TRACE_FILENAME_ENHANCEMENT = 'BW_Trace_8.mat'
NETWORK_TRACE_FILENAME_BASE = 'BW_Trace_4.mat'
VIEWPORT_TRACE_FILENAME1 = 'view_angle_combo_video1.mat'
VIEWPORT_TRACE_FILENAME2 = 'view_trace_fanyi_amsterdam_2D.mat'


class streaming(object):

	def __init__(self):
		self.video_seg_index_BL = BUFFER_BL_INIT
		self.video_seg_index_EL = BUFFER_EL_INIT
		self.network_seg_index_good = 1  # Guanyu
		self.network_seg_index_bad = 1  # Guanyu
		self.remaining_time = 1
# 		self.remaining_time_bad = 1
		self.video_download_timestamp = 0
# 		self.video_download_timestamp_good = 0
# 		self.video_download_timestamp_bad = 0
		self.buffer_size_BL = BUFFER_BL_INIT
		self.buffer_size_EL = BUFFER_EL_INIT
		self.buffer_size_history_good = []
		self.buffer_size_history_bad = []
		self.downloadedPartialVideo_bad = 0
		self.downloadedPartialVideo_good = 0
		self.EVR_BL_Recordset = []
		self.EVR_EL_Recordset = []
		self.video_segment_good = 0
		self.video_segment_bad = 0
		self.GBUFFER_BL_EMPTY_COUNT = 0
		self.GBUFFER_EL_EMPTY_COUNT = 0
		self.video_version_good = 0
		self.video_version_bad = 0
		self.video_segment_index = 0
		self.yaw_predict_value = 0
		self.pitch_predict_value = 0
		self.yaw_predict_value_quan = 0
		self.pitch_predict_value_quan = 0
		self.record_info_good = []
		self.record_info_bad = []        
		self.video_bw_history_good = []  # Guanyu
		self.video_bw_history_bad = []  # Guanyu

	def run(self, network_trace_good, network_trace_bad, yaw_trace, pitch_trace, video_trace, rate_cut):  # Guanyu
		while (self.video_seg_index_BL < VIDEO_LEN): 
#         or \
# 			(self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
			if not self.downloadedPartialVideo_bad:

				if BW_PRED == 1:
					sniff_BW_bad = self.getCurrentBW(network_trace_bad)  # Guanyu
				elif BW_PRED == 2:
					sniff_BW_bad = self.predictBW_bad()

				# self.control_good(rate_cut, sniff_BW)
				self.control_bad()
				# self.video_segment_good = video_trace[self.video_version][self.video_seg_index_EL]
				self.video_segment_bad = video_trace[self.video_version_bad][self.video_seg_index_BL]
# 				self.predict_yaw(yaw_trace)
# 				self.predict_pitch(pitch_trace)
			if not self.downloadedPartialVideo_good:

				if BW_PRED == 1:
					sniff_BW_good = self.getCurrentBW(network_trace_good) #Guanyu
				elif BW_PRED == 2:
					sniff_BW_good = self.predictBW_good()

				self.control_good(rate_cut, sniff_BW_good)
# 				self.control_bad()
				self.video_segment_good = video_trace[self.video_version_good][self.video_seg_index_EL]
# 				self.video_segment_bad = video_trace[self.video_version_bad][self.video_seg_index_BL]
				self.predict_yaw(yaw_trace)
				self.predict_pitch(pitch_trace)
			previous_time = self.video_fetching(network_trace_good, network_trace_bad, rate_cut, yaw_trace, pitch_trace) #Guanyu
            # previous_time = self.video_fetching_good(network_trace_good, rate_cut, yaw_trace, pitch_trace) #Guanyu

			# return
			if not self.downloadedPartialVideo_bad:
				self.record_info_bad = np.append(self.record_info_bad, 					[self.video_download_timestamp, self.buffer_size_BL, self.buffer_size_EL,
					self.yaw_predict_value, yaw_trace[(self.video_seg_index_EL - 1)*VIDEO_FPS-int(VIDEO_FPS/2)],
					self.pitch_predict_value, pitch_trace[(self.video_seg_index_EL - 1)*VIDEO_FPS-int(VIDEO_FPS/2)],
					sniff_BW_bad, previous_time+1])
			# return
			if not self.downloadedPartialVideo_good:
				self.record_info_good = np.append(self.record_info_good, 					[self.video_download_timestamp, self.buffer_size_BL, self.buffer_size_EL,
					self.yaw_predict_value, yaw_trace[(self.video_seg_index_EL - 1)*VIDEO_FPS-int(VIDEO_FPS/2)],
					self.pitch_predict_value, pitch_trace[(self.video_seg_index_EL - 1)*VIDEO_FPS-int(VIDEO_FPS/2)],
					sniff_BW_good, previous_time+1])
# 
# 			self.video_download_timestamp=np.minimum(previous_time_good, previous_time_bad)
		print(self.network_seg_index_good)  #Guanyu
		print(self.network_seg_index_bad)  #Guanyu
		print('Simluation done')
		# print(len(self.EVR_BL_Recordset))	

	def control_good(self, rate_cut, sniff_BW_good): #Guanyu
		# print(self.buffer_size_BL)
		# print(self.video_seg_index_BL)
		# print(self.downloadedPartialVideo)
		if not self.downloadedPartialVideo_good:
# 			if self.buffer_size_BL < Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN :
# 				self.video_version = 0
# 				self.video_segment_index = self.video_seg_index_BL
				# print(self.video_seg_index_BL, self.video_seg_index_EL, int(np.floor(self.video_download_timestamp)))
			if (self.buffer_size_BL >= Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN ) 				or (self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
# 			if self.video_seg_index_EL < VIDEO_LEN :
        
				# PI control logic
				u_p = KP * (self.buffer_size_EL - Q_REF_EL)
				u_i = 0
				if len(self.buffer_size_history_good) != 0:
# 					print(self.buffer_size_history_good)
					for index in range(0, len(self.buffer_size_history_good)):
						u_i +=  KI  * (self.buffer_size_history_good[index][0] - Q_REF_EL)
				
				u = u_i + u_p

				########################
				if self.buffer_size_EL >= 1:
					v = u + 1
					delta_time = 1
				else :
					v = u + np.ceil(self.video_download_timestamp + (10.**-8)) - self.video_download_timestamp
					delta_time = np.ceil(self.video_download_timestamp) - self.video_download_timestamp
				R_hat = v * sniff_BW_good
				###############
				# v = u + 1
				# delta_time = self.buffer_size_EL
				# R_hat = min(v, delta_time) * sniff_BW
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
				self.video_version_good = current_video_version
				# self.video_segment_index = self.video_seg_index_EL
				# print('time:', self.video_download_timestamp)
		# else:
		# 	print("Still download previous segment")
		return 
    
    # Guanyu

	def control_bad(self):
		if not self.downloadedPartialVideo_bad:
			if self.buffer_size_BL < Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN :
				self.video_version_bad = 0
				# self.video_segment_index_bad = self.video_seg_index_BL
                

	def video_fetching(self, network_trace_good, network_trace_bad, rate_cut, yaw_trace, pitch_trace):	
		temp_video_download_timestamp = self.video_download_timestamp
		video_rate = 0 
		if (network_trace_good[self.network_seg_index_good]*self.remaining_time < self.video_segment_good) and 			(network_trace_bad[self.network_seg_index_bad]*self.remaining_time < self.video_segment_bad):#Guanyu
			self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
			self.video_segment_good = self.video_segment_good - network_trace_good[self.network_seg_index_good]*self.remaining_time
			self.video_segment_bad = self.video_segment_bad - network_trace_bad[self.network_seg_index_good]*self.remaining_time
			self.remaining_time = 1
			self.downloadedPartialVideo_good = 1
			self.downloadedPartialVideo_bad = 1
		elif (self.video_segment_good/network_trace_good[self.network_seg_index_good]) < 			(self.video_segment_bad/network_trace_bad[self.network_seg_index_bad]) :#Guanyu
 			self.video_segment_bad = self.video_segment_bad - network_trace_bad[self.network_seg_index_good]*self.remaining_time
			self.downloadedPartialVideo_bad = 1
			if self.video_version_good >= 1: #Guanyu
				if self.buffer_size_EL < ET_MAX_PRED:
					self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version_good, network_trace_good[self.network_seg_index_good], 
											self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS -int(VIDEO_FPS/2)]])
					self.video_download_timestamp += self.video_segment_good/(network_trace_good[self.network_seg_index_good])
					self.remaining_time -= (self.video_download_timestamp - temp_video_download_timestamp)
					# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
					self.downloadedPartialVideo_good = 0
					self.video_seg_index_EL += 1
					self.buffer_size_EL += 1
				else:
					self.remaining_time = 1
					self.downloadedPartialVideo_good = 0
					self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
					self.video_segment_good = 0
					self.flag = 1
				
				# Liyang
				video_rate = rate_cut[self.video_version_good] - rate_cut[0]
				####
			
			if not self.downloadedPartialVideo_good:
				if self.video_segment_good > 0 :
					if len(self.video_bw_history_good) == 0:
						bw = video_rate/self.video_download_timestamp
					else :
						bw = video_rate/(self.video_download_timestamp - self.video_bw_history_good[-1][1]) #Guanyu
					
					self.video_bw_history_good.append([bw, self.video_download_timestamp, self.video_version_good]) #Guanyu
					# print(bw,network_trace[self.network_seg_index], int(np.floor(temp_video_download_timestamp)))
				else:
					self.video_bw_history_good.append([network_trace_good[self.network_seg_index_good], self.video_download_timestamp, -1]) #Guanyu

            
		elif (self.video_segment_good/network_trace_good[self.network_seg_index_good]) > 		(self.video_segment_bad/network_trace_bad[self.network_seg_index_bad]) :#Guanyu
            
			self.video_segment_good = self.video_segment_good - network_trace_good[self.network_seg_index_good]*self.remaining_time
			self.downloadedPartialVideo_good = 1
			if self.video_version_bad == 0:
				self.EVR_BL_Recordset.append([self.video_seg_index_BL, self.video_version_bad, network_trace_bad[self.network_seg_index_bad]]) #Guanyu
				self.video_download_timestamp += self.video_segment_bad/(network_trace_bad[self.network_seg_index_bad])
				self.remaining_time -= (self.video_download_timestamp - temp_video_download_timestamp)
				# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
				self.downloadedPartialVideo_bad = 0
				self.video_seg_index_BL += 1
				self.buffer_size_BL += 1
				video_rate = rate_cut[0]  ### Liyang
           
			if not self.downloadedPartialVideo_bad:
				if self.video_segment_bad > 0 :
					if len(self.video_bw_history_bad) == 0:
						bw = video_rate/self.video_download_timestamp 
					else :
						bw = video_rate/(self.video_download_timestamp - self.video_bw_history_bad[-1][1])
					
						self.video_bw_history_bad.append([bw, self.video_download_timestamp, self.video_version_bad])
					# print(bw,network_trace[self.network_seg_index], int(np.floor(temp_video_download_timestamp)))
				else:
					self.video_bw_history_bad.append([network_trace_bad[self.network_seg_index_bad], self.video_download_timestamp, -1])
 
		self.buffer_size_BL = max(0, self.buffer_size_BL - (self.video_download_timestamp - temp_video_download_timestamp))
		self.buffer_size_EL = max(0, self.buffer_size_EL - (self.video_download_timestamp - temp_video_download_timestamp))
		
		if np.floor(self.video_download_timestamp) != np.floor(temp_video_download_timestamp):
			if self.buffer_size_BL == 0:
				self.GBUFFER_BL_EMPTY_COUNT += 1
				self.video_seg_index_BL = int(np.floor(self.video_download_timestamp))
			if self.buffer_size_EL == 0:
				self.GBUFFER_EL_EMPTY_COUNT += 1
				self.video_seg_index_EL = int(np.floor(self.video_download_timestamp))
			self.buffer_size_history_good.append([self.buffer_size_EL])
			self.network_seg_index_good += 1 #Guanyu
			self.network_seg_index_bad += 1 #Guanyu

		return temp_video_download_timestamp

	def predict_yaw(self, yaw_trace):
		# print(yaw_trace)
		if self.video_download_timestamp < 1:
			self.yaw_predict_value = 360 
		else:
			if not self.downloadedPartialVideo_good and self.video_version_good >= 1:
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
				yaw_predict_idx = self.video_seg_index_EL*VIDEO_FPS + VIDEO_FPS/2
				self.yaw_predict_value = np.round(np.polyval(yaw_predict_model,yaw_predict_idx))
				# print(yaw_predict_value)
				# Adjust yaw_predict value to range from 1 to 360
			else: return
		self.yaw_predict_value %= 360
		if self.yaw_predict_value == 0: self.yaw_predict_value += 360
		
		# quantize yaw predict value to range from 1 to 12
		# different with the value in Fanyi's Matlab source code
		self.yaw_predict_value_quan = int(self.yaw_predict_value / 30)
		if self.yaw_predict_value_quan == 12: self.yaw_predict_value_quan = 0
		self.yaw_predict_value_quan += 1
		# print(yaw_predict_value_quan)
		return

	def predict_pitch(self, pitch_trace):
		if self.video_download_timestamp < 1:
			self.pitch_predict_value = 90
		else:
			if not self.downloadedPartialVideo_good and self.video_version_good >= 1:
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

	def getCurrentBW(self, network_trace): #Guanyu
		if self.network_seg_index == 0 or np.floor(self.video_download_timestamp) == 0:
			return network_trace[0] #Guanyu
		else:
			# print(int(self.video_download_timestamp))
			# print(network_trace)
			return network_trace[int(np.floor(self.video_download_timestamp))]

	def predictBW_good(self):
		if len(self.video_bw_history_good) == 0: #Guanyu
			return INIT_BW
		else:
			# print(int(self.video_download_timestamp))
			# print(network_trace)
			if len(self.video_bw_history_good) < BW_PRED_SAMPLE_SIZE: #Guanyu
				return sum(row[0] for row in self.video_bw_history_good)/len(self.video_bw_history_good) #Guanyu
			else :
				return sum(row[0] for row in self.video_bw_history_good[-BW_PRED_SAMPLE_SIZE:])/BW_PRED_SAMPLE_SIZE #Guanyu

	def predictBW_bad(self):
		if len(self.video_bw_history_bad) == 0: #Guanyu
			return INIT_BW
		else:
			# print(int(self.video_download_timestamp))
			# print(network_trace)
			if len(self.video_bw_history_bad) < BW_PRED_SAMPLE_SIZE: #Guanyu
				return sum(row[0] for row in self.video_bw_history_bad)/len(self.video_bw_history_bad) #Guanyu
			else :
				return sum(row[0] for row in self.video_bw_history_bad[-BW_PRED_SAMPLE_SIZE:])/BW_PRED_SAMPLE_SIZE            

def main():
	network_trace_bad = loadNetworkTrace(NETWORK_TRACE_FILENAME_BASE) #Guanyu
	network_trace_good = loadNetworkTrace(NETWORK_TRACE_FILENAME_ENHANCEMENT) #Guanyu
	yaw_trace, pitch_trace = loadViewportTrace()
	network_pdf_good, pdf_bins_good, pdf_patches_good = plot_pdf(network_trace_good) #Guanyu
	network_cdf_good, cdf_bins_good, cdf_patches_good = plot_cdf(network_trace_good) #Guanyu
	rate_cut = rate_determine(network_cdf_good)
	video_trace = loadVideoTrace(rate_cut)

	streaming_sim = streaming()
	streaming_sim.run(network_trace_good, network_trace_bad, yaw_trace, pitch_trace, video_trace, rate_cut) #Guanyu
	# print(len(streaming_sim.buffer_size_history))
	if IS_DEBUGGING:
		display(streaming_sim.record_info_good, streaming_sim.record_info_bad, streaming_sim.EVR_BL_Recordset, streaming_sim.EVR_EL_Recordset, 			rate_cut, yaw_trace, pitch_trace, network_trace_good, network_trace_bad) #Guanyu
# 		raw_input()

def loadNetworkTrace(NETWORK_TRACE_FILENAME):
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
	rate_cut = [10]
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

def display(record_info_good, record_info_bad, EVR_BL_Recordset, EVR_EL_Recordset, rate_cut, yaw_trace, pitch_trace, network_trace_good, network_trace_bad):
	# print(len(record_info)/9)
	# print(len(EVR_EL_Recordset))
	# print(len(EVR_BL_Recordset))

	display_result_good = record_info_good.reshape(int(len(record_info_good)/9), 9).T
	display_result_bad = record_info_bad.reshape(int(len(record_info_bad)/9), 9).T    
	f = plt.figure(3)
	plt.plot(display_result_bad[0], display_result_bad[1],'r-', label='Base Tier Path Buffer Length')
	plt.plot(display_result_good[0], display_result_good[2],'b-', label='Enhancement Tier Path Buffer Length')
	plt.legend(loc='upper right')
	plt.title('BTP/ETP Buffer Length')
	plt.xlabel('Second')
	plt.ylabel('Second')
	plt.axis([0, 600, 0, 300])

	# plt.ylim(-1.5, 2.0)
	h = plt.figure(5)
	plt.plot(display_result_good[0], display_result_good[3],'b-', label='Predict Viewport (horizontal)')
	plt.plot(display_result_good[0], display_result_good[4],'r-', label='Real Viewport (horizontal)')
	plt.legend(loc='upper right')
	plt.title('Viewport Predict and Real Trace')
	plt.xlabel('Second')
	plt.ylabel('Degree')
	plt.axis([0, 600, 0, 450])

	p = plt.figure(6)
	plt.plot(display_result_good[8], display_result_good[7],'b-',label='Predict Bandwidth')
	plt.plot(range(1,VIDEO_LEN+1), network_trace_good,'r-',label='Real Bandwidth for good path') #Guanyu
	plt.legend(loc='upper right')
	plt.title('Enhancement Tier Path Bandwidth Predict and Real Trance')
	plt.xlabel('Second')
	plt.ylabel('Mbps')
	plt.axis([0, 600, 0, 400])

    # Guanyu
	k = plt.figure(7)
	plt.plot(display_result_bad[8], display_result_bad[7],'b-',label='Predict Bandwidth')
	plt.plot(range(1,VIDEO_LEN+1), network_trace_bad,'r-',label='Real Bandwidth for bad path')
	plt.legend(loc='upper right')
	plt.title('Base Tier Path Bandwidth Predict and Real Trance')
	plt.xlabel('Second')
	plt.ylabel('Mbps')
	plt.axis([0, 600, 0, 400])
    
	display_bitrate = [0]*VIDEO_LEN
	receive_bitrate = [0]*VIDEO_LEN
	# print(EVR_BL_Recordset)
	for i in range (0,BUFFER_BL_INIT):
		display_bitrate[i] += rate_cut[0]
		receive_bitrate[i] += rate_cut[0]
	display_bitrate[0] += rate_cut[3]
	receive_bitrate[0] += rate_cut[3]
	for i in range(0, len(EVR_BL_Recordset)):
		display_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]
		receive_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]

	for i in range(0,len(EVR_EL_Recordset)):
		yaw_distance = 0.
		eff= 0.
		sum_eff = 0.
		for j in range(0, VIDEO_FPS):
			yaw_distance = min(np.abs(yaw_trace[EVR_EL_Recordset[i][0]*VIDEO_FPS+j] - EVR_EL_Recordset[i][3]), 							360 - np.abs(yaw_trace[EVR_EL_Recordset[i][0]*VIDEO_FPS+j] - EVR_EL_Recordset[i][3]))
			eff = min(1, max(0, (105 - yaw_distance)/90))
			sum_eff += eff
		sum_eff /= VIDEO_FPS
		display_bitrate[EVR_EL_Recordset[i][0]] += sum_eff*rate_cut[EVR_EL_Recordset[i][1]]
		receive_bitrate[EVR_EL_Recordset[i][0]] += rate_cut[EVR_EL_Recordset[i][1]]
		# print (sum_eff, EVR_EL_Recordset[i][0])
	
	g = plt.figure(4)
	plt.plot(range(1,VIDEO_LEN+1), display_bitrate, 'b-', label='Effective Video Bitrate')
	plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-', label='Received Video Bitrate')
	plt.legend(loc='upper right')
	plt.title('Effective & Received Video Bitrate')
	plt.xlabel('Second')
	plt.ylabel('Mbps')
	plt.axis([0, 600, 0, 250])

	# i = plt.figure(5)
	# plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-')
	# print(len(display_bitrate))
	# print(EVR_BL_Recordset)
	# print(EVR_EL_Recordset)
	f.show()
	g.show()
	h.show()
	p.show()
	k.show()    
	f.savefig('Base_Tier_Path_&_Enhancement_Tier_Path_Buffer_Length.eps', format='eps', dpi=1000)
	g.savefig('Effective_Received_Video_Bitrate.eps', format='eps', dpi=1000)
	h.savefig('Viewport_Predict_&_Real_Trace.eps', format='eps', dpi=1000)
	p.savefig('Enhancement_Tier_Path_Bandwidth_Predict_&_Real_Trance.eps', format='eps', dpi=1000)
	k.savefig('Base_Tier_Path_Bandwidth_Predict_&_Real_Trance.eps', format='eps', dpi=1000)

	return

if __name__ == '__main__':
	main()

