
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G

# Display buffer initialization
BUFFER_BL_INIT = 10
BUFFER_EL_INIT = 1
Q_REF_BL = 10
Q_REF_EL = 1
ET_MAX_PRED = Q_REF_EL + 1

# Control parameters
IS_SAVING = 0
IS_DEBUGGING = 1
DISPLAYRESULT = 1
TRACE_IDX = 1
KP = 0.6		# P controller
KI = 0.01		# I controller
PI_RANGE = 60
MAX_VIDEO_VERSION = 3

# Video parameters
VIDEO_LEN = 600		# seconds
VIDEO_FPS = 30		# hz	
NETWORK_TRACE_LEN = VIDEO_LEN 		# seconds
VIEW_PRED_SAMPLE_LEN = 30	# samples used for prediction
POLY_ORDER = 1				#  1: linear   2: quadratic
FRAME_MV_LIMIT = 180		# horizontal motion upper bound in degree, used for horizontal circular rotation
FIGURE_NUM = 1
#
INIT_BW = 30
BW_PRED = 2
BW_PRED_SAMPLE_SIZE = 5

VIEWPORT_TRACE_FILENAME = 'view_angle_combo_video1.mat'  ##video 1-4
REGULAR_CHANNEL_TRACE = 'BW_Trace_4.mat'
REGULAR_MULTIPLE = 50
REGULAR_ADD = 10
## version 1: 5G, only correct, no repair
## version 2: 5G, correct and repair
## version 3: 4G, only correct, no repair
## Other versions: 4G, no correct, no repair
STREAMING_VERSION = 1
if STREAMING_VERSION == 1:
	CELL = 1 ## 5G
	NETWORK_TRACE_FILENAME = 'BW_Trace_5G_2.txt'
	EXTRA_MULTIPLE = 1
	EXTRA_ADD = 0
	IS_CORRECT = 1
	IS_REPAIR = 0
	CORRECT_TIME_HEAD = 0.1
	REPAIR_TIME_HEAD = 0.1    ## Useless
	CORRECT_TIME_THRES = 0.3
	REPAIR_TIME_THRES = 0.3

elif STREAMING_VERSION == 2:
	CELL = 1 ## 5G
	NETWORK_TRACE_FILENAME = 'BW_Trace_5G_2.txt'
	EXTRA_MULTIPLE = 1
	EXTRA_ADD = 0	
	IS_CORRECT = 1
	IS_REPAIR = 1
	CORRECT_TIME_HEAD = 0.1
	REPAIR_TIME_HEAD = 0.1
	CORRECT_TIME_THRES = 0.3
	REPAIR_TIME_THRES = 0.3


elif STREAMING_VERSION == 3:
	CELL = 0 ## 4G
	NETWORK_TRACE_FILENAME = 'BW_Trace_1.mat'
	EXTRA_MULTIPLE = 100
	EXTRA_ADD = 0
 	IS_CORRECT = 1
	IS_REPAIR = 0
	CORRECT_TIME_HEAD = 0.9
	REPAIR_TIME_HEAD = 0.1   ## Useless
	CORRECT_TIME_THRES = 0.9
	REPAIR_TIME_THRES = 0.5

else:
	CELL = 0 ## 4G
	NETWORK_TRACE_FILENAME = 'BW_Trace_1.mat'
	EXTRA_MULTIPLE = 100
	EXTRA_ADD = 0
 	IS_CORRECT = 0
	IS_REPAIR = 0
	CORRECT_TIME_HEAD = 0.1  ## Useless
	REPAIR_TIME_HEAD = 0.1   ## Useless
	CORRECT_TIME_THRES = 0.9
	REPAIR_TIME_THRES = 0.5



## RTT
DELAY = 0
RTT_1 = 50.0/1000.0   # 50ms delay
RTT_2 = 5.0/1000.0    # 5ms delay
# RTT_2 = 0.0
DISABLE_RATE = 0.0
RETRAN_UTILI_INIT = 0.0
RETRAN_BW_MEAN = 1000.0
RETRAN_BW_STAND = 100.0
RETRAN_EXTRA = 1.15
COST_EFF = 10  		  ##   


USER_VP = 120.0
VP_SPAN_YAW = 150.0
VP_SPAN_PITCH = 180.0
TILE_SIZE = 30.0
MAX_TILES = int(360.0/TILE_SIZE)
# Directory


class streaming(object):

	def __init__(self):
		self.video_seg_index_BL = BUFFER_BL_INIT 
		self.video_seg_index_EL = BUFFER_EL_INIT
		self.network_seg_index = 0
		self.remaining_time = 1
		self.video_download_timestamp = 0
		self.buffer_size_BL = BUFFER_BL_INIT
		self.buffer_size_EL = BUFFER_EL_INIT
		self.buffer_size_history = []
		self.downloadedPartialVideo = 0
		self.correct_ptr = 0
		self.delay = DELAY
		self.correctness_using = 0
		self.request_pre = 0
		self.is_empty_el = 0
		self.addition_time = 0.0

		self.EVR_BL_Recordset = []
		self.EVR_EL_Recordset = []
		self.video_segment = 0
		self.GBUFFER_BL_EMPTY_COUNT = 0
		self.GBUFFER_EL_EMPTY_COUNT = 0
		self.video_version = 0
		self.video_segment_index = 0
		self.yaw_predict_value = 0
		self.pitch_predict_value = 0
		self.yaw_predict_value_quan = 0
		self.pitch_predict_value_quan = 0
		self.record_info = []
		self.bw_info = []
		self.video_bw_history = []



	def run(self, network_trace, yaw_trace, pitch_trace, video_trace, rate_cut, network_trace_aux):
		while self.video_seg_index_BL < VIDEO_LEN or \
			(self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
			if not self.downloadedPartialVideo:

				if BW_PRED == 1:
					sniff_BW = self.getCurrentBW(network_trace)
				elif BW_PRED == 2:
					sniff_BW = self.predictBW()

				self.bw_info = np.append(self.bw_info, [sniff_BW, self.video_download_timestamp])

				self.video_version, self.video_segment_index = self.control(rate_cut, sniff_BW)
				self.video_segment = video_trace[self.video_version][self.video_segment_index]
				temp_index = self.video_segment_index
				if self.video_version >= 1:
					self.yaw_predict_value, self.yaw_predict_value_quan = self.predict_yaw(yaw_trace, self.video_download_timestamp, self.video_segment_index)
					self.pitch_predict_value, self.pitch_predict_value_quan = self.predict_pitch(pitch_trace, self.video_download_timestamp, self.video_segment_index)


			previous_time, recording_EL = self.video_fetching(network_trace, rate_cut, yaw_trace, pitch_trace, network_trace_aux)
			# return
			## SHOULD MODIFY
			if not self.downloadedPartialVideo and self.video_version >= 1 and recording_EL:
			# 	self.record_info = np.append(self.record_info, \
			# 		[self.video_download_timestamp, self.buffer_size_BL, self.buffer_size_EL,
			# 		temp_yaw, yaw_trace[int((self.video_seg_index_EL - 1)*VIDEO_FPS-VIDEO_FPS/2)],
			# 		temp_pitch, pitch_trace[int((self.video_seg_index_EL - 1)*VIDEO_FPS-VIDEO_FPS/2)],
			# 		sniff_BW, previous_time])
				self.record_info = np.append(self.record_info, \
					[self.video_download_timestamp, self.buffer_size_BL, self.buffer_size_EL,
					self.yaw_predict_value, yaw_trace[int(temp_index*VIDEO_FPS+VIDEO_FPS/2)],
					self.pitch_predict_value, pitch_trace[int(temp_index*VIDEO_FPS+VIDEO_FPS/2)], previous_time])
		# print(self.video_bw_history)
		print(self.network_seg_index)
		print('Simluation done')
		print(len(self.EVR_EL_Recordset))
		print(self.video_download_timestamp)
		print(self.GBUFFER_EL_EMPTY_COUNT)
		print(self.GBUFFER_BL_EMPTY_COUNT)
		print(self.addition_time)
		# print(len(self.EVR_BL_Recordset))	

	def control(self, rate_cut, sniff_BW):
		# print(self.buffer_size_BL)
		# print(self.video_seg_index_BL)
		# print(self.downloadedPartialVideo)
		# if not self.downloadedPartialVideo:
		current_video_version = -1
		video_segment_index = -1
		if self.buffer_size_BL < Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN :
			current_video_version = 0
			video_segment_index = self.video_seg_index_BL
			# print(self.video_seg_index_BL, self.video_seg_index_EL, int(np.floor(self.video_download_timestamp)))
		elif (self.buffer_size_BL >= Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN ) \
			or (self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
			#PI control logic
			u_p = KP * (self.buffer_size_EL - Q_REF_EL)
			u_i = 0
			if len(self.buffer_size_history) != 0:
				# print(self.buffer_size_history)
				# for index in range(0, len(self.buffer_size_history)):
				for index in range(1, min(PI_RANGE+1, len(self.buffer_size_history)+1)):
					if not IS_REPAIR:
						u_i +=  KI  * (self.buffer_size_history[-index][1] - Q_REF_EL)
					else:
						u_i +=  KI  * (self.buffer_size_history[-index][1] - Q_REF_EL)
			
			u = u_i + u_p
			########################
			if self.buffer_size_EL >= 1:
				v = u + 1
				delta_time = 1
			else :
				v = u + np.ceil(self.video_download_timestamp + (10.**-8)) - self.video_download_timestamp
				delta_time = np.ceil(self.video_download_timestamp) - self.video_download_timestamp
			R_hat = v * sniff_BW
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
			# self.video_version = current_video_version
			video_segment_index = self.video_seg_index_EL
			# print('time:', self.video_download_timestamp)

		return current_video_version, video_segment_index

	def video_fetching(self, network_trace, rate_cut, yaw_trace, pitch_trace, network_trace_aux):
		temp_video_download_timestamp = self.video_download_timestamp
		video_rate = 0  #### Liyang
		recording_EL = 0
		is_same_el_index = 0

		if not self.downloadedPartialVideo and self.delay:
			self.request_pre = max(0, RTT_1-self.remaining_time)
			self.video_download_timestamp = min(np.ceil(temp_video_download_timestamp), self.video_download_timestamp + RTT_1)
			self.remaining_time = max(0, self.remaining_time - RTT_1)

			# if self.remaining_time <= 0 or np.floor(temp_video_download_timestamp) != np.floor(self.video_download_timestamp):

		# print(self.video_segment, network_trace[self.network_seg_index]*self.remaining_time)
		# print("fetching network", self.network_seg_index)
		# print("fetching el index", self.video_seg_index_EL)
		# print("fetching version", self.video_version)
		# print("current time", self.video_download_timestamp)
		if network_trace[self.network_seg_index]*self.remaining_time >= self.video_segment:
			if self.video_version == 0:
				self.EVR_BL_Recordset.append([self.video_seg_index_BL, self.video_version, network_trace[self.network_seg_index]])
				self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
				self.remaining_time -= (self.video_download_timestamp - temp_video_download_timestamp)
				# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
				self.downloadedPartialVideo = 0
				self.video_seg_index_BL += 1 
				self.video_seg_index_EL = max(self.video_seg_index_EL, int(np.ceil(self.video_download_timestamp)))
				if self.video_seg_index_BL > int(np.floor(self.video_download_timestamp)):
					self.buffer_size_BL += 1
				# elif self.video_seg_index_BL == int(np.floor(self.video_download_timestamp)):
				# 	self.buffer_size_BL += (np.ceil(self.video_download_timestamp) - self.video_download_timestamp)
				video_rate = rate_cut[0] ### Liyang
				is_same_el_index = 0
			elif self.video_version >= 1:
				# print("buffer el", self.buffer_size_EL)
				if self.buffer_size_EL < ET_MAX_PRED:
					self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
					self.remaining_time -= (self.video_download_timestamp - temp_video_download_timestamp)
					# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
					self.downloadedPartialVideo = 0
					temprol_eff = 1 - min(1, max(self.video_download_timestamp - self.video_seg_index_EL, 0))
					if self.video_seg_index_EL >= int(np.floor(self.video_download_timestamp)):
						self.buffer_size_EL += 1
						if not IS_REPAIR:
							if self.video_seg_index_EL == int(np.floor(self.video_download_timestamp)):
								# self.buffer_size_EL += (np.ceil(self.video_download_timestamp) - self.video_download_timestamp)
								is_same_el_index = 1
							# else:
								# self.buffer_size_EL += 1
							self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
								self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
								self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
								VP_SPAN_YAW, VP_SPAN_PITCH, temprol_eff, RETRAN_UTILI_INIT, 0])
							recording_EL = 1
						else:
							if len(self.EVR_EL_Recordset) > 0:
								if self.EVR_EL_Recordset[-1][0] < self.video_seg_index_EL:
									# self.buffer_size_EL += 1
									self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
										self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
										self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
										VP_SPAN_YAW, VP_SPAN_PITCH, temprol_eff, RETRAN_UTILI_INIT, 0])
									recording_EL = 1
									# print("Regular channel replace 5G chunk!!!!!!--1-1-1-1")
									# print("Append:", self.video_seg_index_EL)
								elif self.EVR_EL_Recordset[-1][0] == self.video_seg_index_EL:
									if temprol_eff > self.EVR_EL_Recordset[-1][9]+0.2 and self.video_version == MAX_VIDEO_VERSION:
										self.EVR_EL_Recordset[-1][3] = self.yaw_predict_value_quan*30-15
										self.EVR_EL_Recordset[-1][5] = self.pitch_predict_value_quan
										self.EVR_EL_Recordset[-1][9] = temprol_eff
										# self.buffer_size_EL += 1
										recording_EL = 1
										# print("Regular channel replace 5G chunk!!!!!!---")
										# print("update existing:", self.video_seg_index_EL)
									else:
										print("Arrive at same sec, but worse", self.video_seg_index_EL)

								else:
									print("arrive too late", self.video_seg_index_EL)
									print("Shell never been triggered")
									assert(1==0)
									is_same_el_index = -1
								if self.video_seg_index_EL == int(np.floor(self.video_download_timestamp)):
									is_same_el_index = 1
							else:
								self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
									self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
									self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
									VP_SPAN_YAW, VP_SPAN_PITCH, temprol_eff, RETRAN_UTILI_INIT, 0])
								recording_EL = 1

					else:
						is_same_el_index = -1
					self.video_seg_index_EL = max(self.video_seg_index_EL + 1, int(np.ceil(self.video_download_timestamp)))
					# print("el change:", self.video_seg_index_EL)
				else:
					self.remaining_time = 1
					self.downloadedPartialVideo = 0
					self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
					self.video_segment = 0
					self.video_version = 0
				
				#### Liyang
				video_rate = rate_cut[self.video_version] - rate_cut[0]
				####
			else :
				print("Unknown video version.")
				self.remaining_time = 1
				self.downloadedPartialVideo = 0
				self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
				self.video_segment = 0
				self.video_version = 0
			
			if not self.downloadedPartialVideo:
				if self.video_segment > 0 :
					if len(self.video_bw_history) == 0:
						bw = video_rate/self.video_download_timestamp 
					else :
						bw = video_rate/(self.video_download_timestamp - self.video_bw_history[-1][1])
					
					self.video_bw_history.append([bw, self.video_download_timestamp, self.video_version])
					# print(bw,network_trace[self.network_seg_index], int(np.floor(temp_video_download_timestamp)))
				else:
					self.video_bw_history.append([network_trace[self.network_seg_index], self.video_download_timestamp, -1])

		else:
			# if self.video_version == 0:
			# 	print('Download base tier, bandwidth is not enough.')
			# elif self.video_version >= 1:
			# 	print('Download enhancement tier, bandwidth is not enough.')
			self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8)) + self.request_pre
			self.video_segment = self.video_segment - network_trace[self.network_seg_index]*self.remaining_time
			self.remaining_time = 1 - self.request_pre
			self.downloadedPartialVideo = 1
			self.request_pre = 0
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
		if not self.downloadedPartialVideo:
			if is_same_el_index  == 0 : ## download one is useful
				if not self.is_empty_el:
					self.buffer_size_EL = max(0, self.buffer_size_EL - (self.video_download_timestamp - temp_video_download_timestamp))
			elif is_same_el_index == 1:
				self.buffer_size_EL = max(0, self.buffer_size_EL - (self.video_download_timestamp - np.floor(temp_video_download_timestamp)))
			else:
				assert(np.round(self.buffer_size_EL == 0))
		else:
			self.buffer_size_EL = max(0, self.buffer_size_EL - (self.video_download_timestamp - temp_video_download_timestamp))
		
		if self.buffer_size_EL == 0:
			self.is_empty_el = 1
		else:
			self.is_empty_el = 0
		# print("Recording buffer %f %f %f %f %f %f" % (self.video_download_timestamp,  temp_video_download_timestamp, self.buffer_size_EL, self.buffer_size_BL, is_same_el_index, self.video_seg_index_EL))
		if np.floor(self.video_download_timestamp) != np.floor(temp_video_download_timestamp):
			if self.buffer_size_BL == 0:
				self.GBUFFER_BL_EMPTY_COUNT += 1
				self.video_seg_index_BL = int(np.floor(self.video_download_timestamp)) ## start to fast retransmit BL video
			if self.buffer_size_EL == 0:
				self.GBUFFER_EL_EMPTY_COUNT += 1
				# self.video_seg_index_EL = int(np.floor(self.video_download_timestamp))
			self.buffer_size_history.append([self.buffer_size_BL, self.buffer_size_EL])
			self.network_seg_index += 1

			if self.retransmit_available(network_trace_aux) and IS_CORRECT:
				self.correct(yaw_trace, pitch_trace, rate_cut, network_trace_aux)

		return temp_video_download_timestamp, recording_EL


	## Check 5G availability
	def retransmit_available(self, network_trace_aux):
		
		availability = network_trace_aux[int(np.floor(self.video_download_timestamp))]
		if availability != 0:
			return True
		else:
			return False

	def repair(self, correct_time, correct_index, yaw_trace, pitch_trace, rate_cut, network_trace_aux):
		if self.correctness_using == 0 and correct_index < 600:	
			# print("Repair:", correct_index)	
			correct_time -= REPAIR_TIME_HEAD
			new_yaw_predict_value, new_yaw_predict_value_quan = self.predict_yaw(yaw_trace, correct_time, correct_index)
			new_pitch_predict_value, new_pitch_predict_value_quan = self.predict_pitch(pitch_trace, correct_time, correct_index)
			## Retransmit the how chunk as previous 

			extra_segment = rate_cut[MAX_VIDEO_VERSION]-rate_cut[0]
			downloading_5G = 0
			new_temprol_eff = 0
			## Start to retransmit
			# retran_bandwidth = np.random.normal(RETRAN_BW_MEAN, RETRAN_BW_STAND)
			retran_bandwidth_1 = network_trace_aux[correct_index-1]
			retran_bandwidth_2 = network_trace_aux[correct_index]
			self.correctness_using = 1
			## Calculate transmitting time, shall be replaced by function with real trace
			if extra_segment > retran_bandwidth_1*REPAIR_TIME_HEAD:
				if retran_bandwidth_2 == 0:
					print("5G Too bad")
					self.correctness_using = 0
					return					
				downloading_5G = (extra_segment - retran_bandwidth_1*REPAIR_TIME_HEAD)/retran_bandwidth_2
				if downloading_5G > REPAIR_TIME_THRES:
					print("5G Too bad")
					self.correctness_using = 0
					self.addition_time += (REPAIR_TIME_THRES + REPAIR_TIME_HEAD)
					return
				new_temprol_eff = min(1, 1 - downloading_5G)
				self.addition_time += (downloading_5G + REPAIR_TIME_HEAD)
			else:
				new_temprol_eff = 1
				self.addition_time += extra_segment/retran_bandwidth_1
			if len(self.EVR_EL_Recordset) > 0:
				assert self.EVR_EL_Recordset[-1][0] < correct_index
			# self.buffer_size_EL += 1

			self.EVR_EL_Recordset.append([correct_index, MAX_VIDEO_VERSION, (retran_bandwidth_1+retran_bandwidth_2)/2, \
				new_yaw_predict_value_quan*30-15, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
				new_pitch_predict_value_quan, pitch_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
				VP_SPAN_YAW, VP_SPAN_PITCH, new_temprol_eff, extra_segment, 1])
			# print("repair degree",new_yaw_predict_value_quan*30-15, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2])
			# print("Effective time", new_temprol_eff)

			self.record_info = np.append(self.record_info, \
					[self.video_download_timestamp, self.buffer_size_BL, self.buffer_size_EL,
					new_yaw_predict_value, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
					new_pitch_predict_value, pitch_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2], correct_time])
			self.correctness_using = 0
		return

	## To use the second 5G high-speed low-latency connection to correct vp
	def correct(self,yaw_trace, pitch_trace, rate_cut, network_trace_aux):
		if self.correctness_using == 0:
			correct_time = np.floor(self.video_download_timestamp)
			correct_index = int(np.floor(self.video_download_timestamp))
			next_EL = []
			while self.correct_ptr < len(self.EVR_EL_Recordset) and self.EVR_EL_Recordset[self.correct_ptr][0] < correct_index :
				self.correct_ptr += 1
			if self.correct_ptr >= len(self.EVR_EL_Recordset):
				if IS_REPAIR and correct_index < VIDEO_LEN:
					# print("Repair:", correct_index)
					# if len(self.EVR_EL_Recordset) > 0:
					# 	print("last one",self.EVR_EL_Recordset[-1][0])
					self.repair(correct_time, correct_index, yaw_trace, pitch_trace, rate_cut, network_trace_aux)
				return
			else:
				correct_time -= CORRECT_TIME_HEAD
				next_EL = self.EVR_EL_Recordset[self.correct_ptr]
				assert (next_EL[9] == 1)
				assert (next_EL[10] == 0)
				# print(next_EL[0])
				if next_EL[0] > correct_index:
					# No need to correct
					## Maybe special case, wrong!! Pay attention
					print("Too far away, not yet.")
					print("Must be handled!")
					return
				elif next_EL[0] == correct_index:
					# print("correct chunk:", correct_index)
					direction_central = 0
					new_yaw_predict_value, new_yaw_predict_value_quan = self.predict_yaw(yaw_trace, correct_time, next_EL[0])
					new_pitch_predict_value, new_pitch_predict_value_quan = self.predict_pitch(pitch_trace, correct_time, next_EL[0])
					new_left_boundary = new_yaw_predict_value - USER_VP/2
					new_right_boundary = new_yaw_predict_value + USER_VP/2
					if new_left_boundary < 0:
						new_left_boundary += 360
					if new_right_boundary >= 360:
						new_right_boundary -= 360
					central = next_EL[3]   # Value recorded in EL Recordest
					left_boundary = central - VP_SPAN_YAW/2
					right_boundary = central + VP_SPAN_YAW/2
					if left_boundary < 0:
						left_boundary += 360
					if right_boundary >= 360:
						right_boundary -= 360

					distance = min(np.abs(central - new_yaw_predict_value), 360 - np.abs(central - new_yaw_predict_value))
					if distance <= 15:
						# print("overlap")
						return 
					else:
						## Need to do correctness, has area blank
						## Judge the direction between the existing VP and new predict VP
						area = []  ## area shall be filled
						direction_central = self.decide_direction(central, new_yaw_predict_value)
						if direction_central == 1:
							# print("right_boundary", right_boundary)
							# print("new_right_boundary", new_right_boundary)
							assert (self.decide_direction(right_boundary, new_right_boundary) == 1)
							if self.decide_direction(right_boundary, new_left_boundary == -1):
								area = [right_boundary, new_right_boundary, 1, True]
							else:
								area = [new_left_boundary, new_right_boundary, 1, False]
								print("totally wrong")
						else:
							assert (self.decide_direction(left_boundary, new_left_boundary) == -1)
							# print("right_boundary", left_boundary)
							# print("new_right_boundary", new_left_boundary)
							if self.decide_direction(left_boundary, new_right_boundary == 1):
								area = [new_left_boundary, left_boundary, -1, True]
							else:
								area = [new_left_boundary, new_right_boundary, -1, False]
								print("totally wrong")
						# print("makeup area", area)
						# print("-----------")
						start_tile, end_tile, total_tiles = self.calculate_tiles(area)
						assert (next_EL[1] >= 1)

						## Calculate size of data should be retransmitted
						retran_data_size = (RETRAN_EXTRA*total_tiles/(VP_SPAN_YAW/TILE_SIZE))*(rate_cut[next_EL[1]]-rate_cut[0])
						# print(retran_data_size)

						## Simulate 5G retransmisstion and correct
						downloading_5G = 0
						## Start to retransmit
						retran_bandwidth_1 = network_trace_aux[correct_index-1]
						retran_bandwidth_2 = network_trace_aux[correct_index]
						self.correctness_using = 1
						## Calculate transmitting time, shall be replaced by function with real trace
						
						if retran_data_size > retran_bandwidth_1*CORRECT_TIME_HEAD:
							downloading_5G = (retran_data_size - retran_bandwidth_1*CORRECT_TIME_HEAD)/retran_bandwidth_2
							if downloading_5G >= CORRECT_TIME_THRES:
								self.EVR_EL_Recordset[self.correct_ptr][10] = retran_data_size*CORRECT_TIME_THRES
								self.correctness_using = 0
								self.addition_time += (CORRECT_TIME_THRES + CORRECT_TIME_HEAD)
								print("Second connection too bad!!!")
								return
						new_temprol_eff = 1 - downloading_5G
						self.addition_time += (downloading_5G + CORRECT_TIME_HEAD)
						new_yaw_span = 0.0
						new_central = 0.0
						## If continueous
						if area[3]:
							new_distance = min(np.abs(area[0] - area[1]), 360 - np.abs(area[0] - area[1]))
							new_yaw_span = VP_SPAN_YAW + new_distance
							if area[2] == 1:
								new_central = next_EL[3] + new_distance/2
							else:
								new_central = next_EL[3] - new_distance/2
							if new_central >= 360.0:
								new_central -= 360.0
							if new_central < 0:
								new_central += 360.0
							##udpate EL recordest
							self.EVR_EL_Recordset[self.correct_ptr][3] = new_central
							self.EVR_EL_Recordset[self.correct_ptr][7] = new_yaw_span
							if distance >= 60.0 and new_temprol_eff <= 0.1:
								self.EVR_EL_Recordset[self.correct_ptr][9] = new_temprol_eff
							self.EVR_EL_Recordset[self.correct_ptr][10] = retran_data_size
							self.correctness_using = 0
							print("Update EL", new_central, new_yaw_span, new_temprol_eff, retran_data_size, self.EVR_EL_Recordset[self.correct_ptr][10])
							print("area:", distance, area)
						else:
							## Liyang, should modify to update EL recordest
							return
							## 5G is not usable
						# print("Could correct!")
						# print("previous central", central)
						# print("new central", new_yaw_predict_value)
						# print("direction", direction_central)
						# print("------------------------------")
		else:
			## 5G is being used
			print("Correct function is not available right now.")

		return 


	def decide_direction(self, degree_1, degree_2):
		direction = 0
		if np.abs(degree_1 - degree_2) <= 360 - np.abs(degree_1 - degree_2):
			if degree_1 >= degree_2:
				direction = -1
			else:
				direction = 1
		else:
			if degree_1 >= degree_2:
				direction = 1
			else:
				direction = -1
		return direction

	def calculate_tiles(self, area):
		assert (len(area) == 4)
		assert (area[0] != area[1])
		total_tiles = 0
		first_tile = int(area[0]/TILE_SIZE)
		last_tile = 0
		if area[1] == 0:
			last_tile = MAX_TILES - 1
		else:
			last_tile = int((area[1]-0.5)/TILE_SIZE)
		if first_tile <= last_tile:
			total_tiles = last_tile - first_tile + 1
		else:
			total_tiles = MAX_TILES - first_tile + last_tile + 1
		# print(area, total_tiles)
		return first_tile, last_tile, total_tiles

	def predict_yaw(self, yaw_trace, video_download_timestamp,video_segment_index):
		# print(yaw_trace)
		yaw_predict_value = 0
		yaw_predict_value_quan = 0
		if video_download_timestamp < 1:
			yaw_predict_value = 360 
		else:
			vp_index = np.arange(-VIEW_PRED_SAMPLE_LEN,0) + int(video_download_timestamp*VIDEO_FPS)
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
			yaw_predict_idx = int(video_segment_index*VIDEO_FPS + VIDEO_FPS/2)
			yaw_predict_value = np.round(np.polyval(yaw_predict_model,yaw_predict_idx))
			# print(yaw_predict_value)
			# Adjust yaw_predict value to range from 1 to 360
		yaw_predict_value %= 360
		# if self.yaw_predict_value == 0: self.yaw_predict_value += 360
		
		# quantize yaw predict value to range from 1 to 12
		# different with the value in Fanyi's Matlab source code
		yaw_predict_value_quan = int(yaw_predict_value / 30)
		if yaw_predict_value_quan == 12: yaw_predict_value_quan = 0
		yaw_predict_value_quan += 1
		# print(yaw_predict_value_quan)
		return yaw_predict_value, yaw_predict_value_quan

	def predict_pitch(self, pitch_trace, video_download_timestamp, video_segment_index):
		pitch_predict_value = 0
		pitch_predict_value_quan = 0
		if video_download_timestamp < 1:
			pitch_predict_value = 90
		else:
			# print(self.video_download_timestamp)
			vp_index = np.arange(-VIEW_PRED_SAMPLE_LEN,0) + int(np.floor(video_download_timestamp*VIDEO_FPS))
			vp_value = []
			for index in vp_index:
				vp_value.append(pitch_trace[index])
			pitch_predict_model = np.polyfit(vp_index, vp_value, POLY_ORDER)
			pitch_predict_idx = int(video_segment_index*VIDEO_FPS - VIDEO_FPS/2)
			pitch_predict_value = np.round(np.polyval(pitch_predict_model, pitch_predict_idx))

		if pitch_predict_value in range(0, 46):
			pitch_predict_value_quan = 1
		elif pitch_predict_value in range(46, 91):
			pitch_predict_value_quan = 2
		elif pitch_predict_value in range(91, 136):
			pitch_predict_value_quan = 3
		else:
			pitch_predict_value_quan = 4
		return pitch_predict_value, pitch_predict_value_quan

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
	network_trace = loadNetworkTrace(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	# print(min(network_trace))
	network_trace_aux = []
	if CELL == 0: ## 4G trace
		network_trace_aux = loadNetworkTrace(NETWORK_TRACE_FILENAME, EXTRA_MULTIPLE, EXTRA_ADD)
	elif CELL == 1: ## 5G trace
		network_trace_aux = load_5G.load_5G_Data(NETWORK_TRACE_FILENAME, EXTRA_MULTIPLE, EXTRA_ADD)
	else:
		print("Load BW trace error!!")
	# print(min(network_trace_aux))
	yaw_trace, pitch_trace = loadViewportTrace()
	network_pdf, pdf_bins, pdf_patches = plot_pdf(network_trace)
	network_cdf, cdf_bins, cdf_patches = plot_cdf(network_trace)
	extra_network_pdf, extra_pdf_bins, extra_pdf_patches = plot_pdf(network_trace_aux)
	extra_network_cdf, extra_cdf_bins, extra_cdf_patches = plot_cdf(network_trace_aux)
	rate_cut = rate_determine(network_cdf)
	video_trace = loadVideoTrace(rate_cut)
	streaming_sim = streaming()
	streaming_sim.run(network_trace, yaw_trace, pitch_trace, video_trace, rate_cut, network_trace_aux)
	# print(len(streaming_sim.buffer_size_history))
	if IS_DEBUGGING:
		display(streaming_sim.record_info, streaming_sim.EVR_BL_Recordset, streaming_sim.EVR_EL_Recordset, \
			rate_cut, yaw_trace, pitch_trace, network_trace, streaming_sim.bw_info, streaming_sim.buffer_size_history, network_trace_aux)
		raw_input()

def loadNetworkTrace(filename, multiple, addition):
	mat_contents = sio.loadmat(filename)
	trace_data = multiple*mat_contents['delta_bit_debugging_buffer'] # array of structures
	# print(type(trace_data), len(trace_data))
	assert (len(trace_data) > NETWORK_TRACE_LEN)
	result = []
	for x in range(0, NETWORK_TRACE_LEN):
		result.append(trace_data[x][0])
	# print(result, len(result))
	result = [x+addition for x in result]
	# print(result, len(result))
	return result

def loadViewportTrace():
	mat_contents = sio.loadmat(VIEWPORT_TRACE_FILENAME)
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
	global FIGURE_NUM
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	n, bins, patches = plt.hist(trace, range(0, int(np.ceil(max(trace))) + 1), normed = 1, label='PDF', color='b')
	plt.title('Bandwidth Distribution')
	plt.xlabel('Mbps', fontsize=15)
	# plt.ylabel('', fontsize=15)
	plt.tick_params(axis='both', which='major', labelsize=15)
	# plt.tick_params(axis='both', which='minor', labelsize=15)
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def plot_cdf(trace):
	global FIGURE_NUM
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM +=1
	n, bins, patches = plt.hist(trace, range(0,int(np.ceil(max(trace))) + 1), normed = 1, cumulative=True, label='CDF', histtype='stepfilled', color='b')
	# print(a,b,c)
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def rate_determine(network_cdf):
	cut_percent = [0.4, 0.6, 0.8]
	rate_cut = [10.0]
	cut_idx = 0
	for cum in network_cdf:
		if cum >= np.max(network_cdf) * cut_percent[cut_idx]:
			rate_cut.append(np.round(network_cdf.tolist().index(cum)))
			cut_idx += 1
			if cut_idx >= len(cut_percent): 
				break
	if IS_DEBUGGING: 
		print('Rate0 = %f, Rate1 = %f, Rate2 = %f, Rate3 = %f' % (rate_cut[0],rate_cut[1],rate_cut[2],rate_cut[3]))
	return rate_cut

def display(record_info, EVR_BL_Recordset, EVR_EL_Recordset, rate_cut, yaw_trace, pitch_trace, network_trace, bw_info, buffer_size_history, network_trace_aux):
	# print(len(record_info)/9)
	# print(len(EVR_EL_Recordset))
	# print(len(EVR_BL_Recordset))

	display_result = record_info.reshape(len(record_info)/8, 8).T
	bw_result = bw_info.reshape(len(bw_info)/2, 2).T
	
	global FIGURE_NUM
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM +=1
	# plt.plot(display_result[0], display_result[1],'r+-', markersize = 5, markeredgecolor = 'red',  label='Base Tier Buffer Length')
	# plt.plot(display_result[0], display_result[2],'bo-', markersize = 5, markeredgecolor = 'blue',  label='Enhancement Tier Buffer Length')
	plt.plot(range(0,len(buffer_size_history)), np.array(buffer_size_history).T[0],'r+-', markersize = 6, markeredgewidth = 0.5, markeredgecolor = 'red',  label='Base Tier Buffer Length')
	plt.plot(range(0,len(buffer_size_history)), np.array(buffer_size_history).T[1],'bo-', markersize = 2, markeredgewidth = 0.5, markeredgecolor = 'blue',  label='Enhancement Tier Buffer Length')
	plt.legend(loc='upper right')
	plt.title('BT/ET Buffer Length')
	plt.xlabel('Second')
	plt.ylabel('Second')
	plt.axis([0, 600, 0, 15])

	# plt.ylim(-1.5, 2.0)
	# global FIGURE_NUM
	h = plt.figure(FIGURE_NUM)
	FIGURE_NUM +=1
	plt.plot(display_result[7], display_result[3],'b-', label='Predict Viewport (horizontal central)')
	plt.plot(display_result[7], display_result[4],'r-', label='Real Viewport (horizontal)')
	# print(len(np.arange(0,VIDEO_LEN,1.0/float(VIDEO_FPS))))
	# print(len(yaw_trace[:VIDEO_LEN*VIDEO_FPS]))
	# plt.plot(np.arange(0,VIDEO_LEN, 1.0/VIDEO_FPS), yaw_trace[:VIDEO_LEN*VIDEO_FPS],'r-', label='Real Viewport (horizontal central)')
	plt.legend(loc='upper right')
	plt.title('Viewport Predict and Real Trace')
	plt.xlabel('Second', fontsize=15)
	plt.ylabel('Degree', fontsize=15)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=15)
	plt.axis([0, 600, 0, 450])

	p = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Real Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right')
	plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Bandwidth Trace')
	plt.xlabel('Second', fontsize=15)
	plt.ylabel('Mbps', fontsize=15)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=15)
	plt.axis([0, 600, 0, max(network_trace)+50])
	# plt.axis([0, 600, 0, max(network_trace_aux)+50])


	display_bitrate = [0.0]*VIDEO_LEN
	receive_bitrate = [0.0]*VIDEO_LEN
	extra_cost = [[0.0, 0]]*VIDEO_LEN
	extra_repair = [[0.0, 0]]*VIDEO_LEN
	# print(EVR_BL_Recordset)
	for i in range (0,BUFFER_BL_INIT):
		display_bitrate[i] += (rate_cut[0]/6)
		receive_bitrate[i] += (rate_cut[0]/6)
	display_bitrate[0] += rate_cut[3]
	receive_bitrate[0] += rate_cut[3]
	for i in range(0, len(EVR_BL_Recordset)):
		display_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]/6
		receive_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]/6
	total_correction = 0.0
	total_repair = 0.0
	# print(EVR_EL_Recordset)
	for i in range(0,len(EVR_EL_Recordset)):
		if EVR_EL_Recordset[i][9] != 0:
			yaw_distance = 0.0
			eff= 0.0
			sum_eff = 0.0
			start_frame = int((1-EVR_EL_Recordset[i][9])*30)
			for j in range(start_frame, VIDEO_FPS):
				yaw_distance = min(np.abs(yaw_trace[EVR_EL_Recordset[i][0]*VIDEO_FPS+j] - (EVR_EL_Recordset[i][3])), \
								360 - np.abs(yaw_trace[EVR_EL_Recordset[i][0]*VIDEO_FPS+j] - (EVR_EL_Recordset[i][3])))
				eff = min(1, max(0, (((EVR_EL_Recordset[i][7] + USER_VP)/2) - yaw_distance)/USER_VP))
				sum_eff += eff
			sum_eff /= (VIDEO_FPS - start_frame)
		else:
			sum_eff = 0
		display_bitrate[EVR_EL_Recordset[i][0]] += EVR_EL_Recordset[i][9]*sum_eff*rate_cut[EVR_EL_Recordset[i][1]] ##+ (1-EVR_EL_Recordset[i][9])*rate_cut[0]/6
		receive_bitrate[EVR_EL_Recordset[i][0]] += rate_cut[EVR_EL_Recordset[i][1]]
		if EVR_EL_Recordset[i][11] == 0:	
			extra_cost[EVR_EL_Recordset[i][0]] = [EVR_EL_Recordset[i][10]/8, EVR_EL_Recordset[i][0]]
			# extra_cost[EVR_EL_Recordset[i][0]][1] = EVR_EL_Recordset[i][0]
			total_correction += EVR_EL_Recordset[i][10]
		else:
			assert(EVR_EL_Recordset[i][11] == 1)
			extra_repair[EVR_EL_Recordset[i][0]] = [EVR_EL_Recordset[i][10]/8, EVR_EL_Recordset[i][0]]
			total_repair += EVR_EL_Recordset[i][10]
			# extra_repair[EVR_EL_Recordset[i][0]][1] = EVR_EL_Recordset[i][0]
	# print(extra_cost)
	extra_cost = np.array(extra_cost).T
	extra_repair = np.array(extra_repair).T
	# total_correction = np.sum(extra_cost)
	# total_repair = np.sum(extra_repair)
	print("Total cost data: %f" % (total_correction))
	print("Total repair data: %f" % (total_repair))
	print("Total extra data: %f" % (total_repair+total_correction))

		# print (sum_eff, EVR_EL_Recordset[i][0])
		# print(EVR_EL_Recordset[i][3]*30-15, EVR_EL_Recordset[i][3], EVR_EL_Recordset[i][5],EVR_EL_Recordset[i][9])
	# return
	print("Effective bitrate:", sum(display_bitrate))
	g = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	plt.plot(range(1,VIDEO_LEN+1), display_bitrate, 'bo-', markersize = 3, markeredgewidth = 0.5, markeredgecolor = 'blue', linewidth = 0.75, label='Effective Video Bitrate')
	plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r+-', markersize = 8, markeredgewidth = 0.5, markeredgecolor = 'red', linewidth = 0.75, label='Received Video Bitrate')
	plt.legend(loc='upper right')
	plt.title('Effective & Received Video Bitrate')
	plt.xlabel('Second', fontsize =15)
	plt.ylabel('Mbps',fontsize = 15)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=15)
	plt.axis([0, 600, 0, max(receive_bitrate)+20])

	if IS_CORRECT:
		q = plt.figure(FIGURE_NUM)
		FIGURE_NUM += 1
		plt.bar(extra_cost[1], extra_cost[0], color='red', label='Correction Data', edgecolor = "red")
		plt.bar(extra_repair[1], extra_repair[0], color='blue', label='Retransmission Data', edgecolor = "blue")
		# plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-', label='Received Video Bitrate')
		plt.legend(loc='best')
		plt.title('Extra Data Cost')
		plt.xlabel('Second')
		plt.ylabel('MB')
		plt.axis([0, 600, 0, max(max(extra_cost[0])+1, max(extra_repair[0])+1)])

	# i = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	# plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-')
	# print(len(display_bitrate))
	# print(EVR_BL_Recordset)
	# print(EVR_EL_Recordset)
	f.show()
	g.show()
	h.show()
	p.show()
	if IS_CORRECT:
		q.show()
	if IS_SAVING:
		f.savefig('Two_Tier_BT_&_ET_Buffer_Length.eps', format='eps', dpi=1000)
		g.savefig('Two_Tier_Effective_Received_Video_Bitrate.eps', format='eps', dpi=1000)
		h.savefig('Two_Tier_Viewport_Predict_&_Real_Trace.eps', format='eps', dpi=1000)
		p.savefig('Two_Tier_Bandwidth_Predict_&_Real_Trance.eps', format='eps', dpi=1000)
		q.savefig('Two_Tier_Extra_Cost.eps', format='eps', dpi=1000)

	return

if __name__ == '__main__':
	main()
	
