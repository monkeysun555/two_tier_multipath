
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G
import math

# Display buffer initialization
BUFFER_BL_INIT = 10
BUFFER_EL_INIT = 1
Q_REF_BL = 10
Q_REF_EL = 1
ET_MAX_PRED = Q_REF_EL + 1
AVE_RATIO = 1  # previous 0.8

NORMAL_USING_TRUN = 0
USING_TRUN = 1
# Control parameters
IS_SAVING = 0
IS_DEBUGGING = 1
DISPLAYRESULT = 1
TRACE_IDX = 1
KP = 0.6		# P controller
KI = 0.01		# I controller
PI_RANGE = 60
MAX_VIDEO_VERSION = 3
BITRATE_LEN = 4
# Video parameters
VIDEO_LEN = 300		# seconds
VIDEO_FPS = 30		# hz	
NETWORK_TRACE_LEN = VIDEO_LEN 		# seconds
VIEW_PRED_SAMPLE_LEN = 30	# samples used for prediction
POLY_ORDER = 1				#  1: linear   2: quadratic
FRAME_MV_LIMIT = 180		# horizontal motion upper bound in degree, used for horizontal circular rotation
FIGURE_NUM = 1
#
INIT_BW = 30
BW_PRED = 2
BW_PRED_SAMPLE_SIZE = 10

# ########
# 	 11: 3 9
# 	 12: 3 13
#	 21: 1 9
#	 22: 1 13
#	 31: 2 9
#	 32: 2 13
#######
R_MIN = 100.0
VER = 12
VIEWPORT_TRACE_FILENAME = 'view_angle_combo_video1.mat'  ##video 1-4
VIEWPORT_TRACE_FILENAME_NEW = 'Video_13_alpha_beta.mat'
REGULAR_CHANNEL_TRACE = 'BW_Trace_5G_3.txt'  # 1: partially disturbed  2: unstable  3: stable   4: medium_liyang 5:medium_fanyi
DELAY_TRACE = 'delay_1.txt'
REGULAR_MULTIPLE = 1
REGULAR_ADD = 0
## In this file, alwayes use 5G trace to do streaming, 
## Other versions: 4G, no correct, no repair
CORRECT_TIME_HEAD = 0.1  ## Useless
CORRECT_TIME_THRES = 0.2
											 ##	
#########################################   ##
IS_AGAIN = 1						       ## ##################
#########################################   ##
AGAIN_THRES = 0.9	 						 ##	
AGAIN_BL_THRES = 10


## RTT
DELAY = 0
RTT_1 = 50.0/1000.0   # 50ms delay
RTT_2 = 5.0/1000.0    # 5ms delay
# RTT_2 = 0.0
DISABLE_RATE = 0.0
RETRAN_UTILI_INIT = 0.0
RETRAN_BW_MEAN = 1000.0
RETRAN_BW_STAND = 100.0
COST_EFF = 10  		  ##   


USER_VP = 120.0
VP_SPAN_YAW = 150.0
VP_SPAN_PITCH = 180.0
TILE_SIZE = 15.0

## Extra coding cost
if TILE_SIZE == 15.0:
	RETRAN_EXTRA = 1.2
elif TILE_SIZE == 30.0:
	RETRAN_EXTRA = 1.1
elif TILE_SIZE == 60.0:
	RETRAN_EXTRA = 1.05
else:
	assert 0 == 1
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
		self.addition_data = 0.0
		self.addition_useful_data = 0.0
		self.already_correct = False

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

				if not self.already_correct:
					self.video_version, self.video_segment_index = self.control(rate_cut, sniff_BW, True)
				else:
					self.video_version, self.video_segment_index = self.control(rate_cut, sniff_BW, False)
				# print("Version:", self.video_version)
				if IS_AGAIN:
					if self.video_version >= 0:
						self.video_segment = video_trace[self.video_version][self.video_segment_index]
						# print("No correction!")
					else:
						## Check whether next EL is existing
						downlading = self.correct(yaw_trace, pitch_trace, rate_cut, network_trace)
						if downlading:
							# print("Correction!!!!!!!!")
							self.already_correct = True
							continue
						else:
							self.video_version, self.video_segment_index = self.control(rate_cut, sniff_BW, False)
							self.video_segment = video_trace[self.video_version][self.video_segment_index]
				else:
					self.video_segment = video_trace[self.video_version][self.video_segment_index]

				# print("Following no correction")
				temp_index = self.video_segment_index
				if self.video_version >= 1 or self.video_version == -1:
					assert self.video_version != -1
					if NORMAL_USING_TRUN:
						self.yaw_predict_value, self.yaw_predict_value_quan = self.predict_yaw_trun(yaw_trace, self.video_download_timestamp, self.video_segment_index)
					else:
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
		print(self.addition_data)
		print(self.addition_useful_data)
		# print(len(self.EVR_BL_Recordset))	

	def control(self, rate_cut, sniff_BW, do_correction):
		# print(self.buffer_size_BL)
		# print(self.video_seg_index_BL)
		# print(self.downloadedPartialVideo)
		# if not self.downloadedPartialVideo:
		current_video_version = -1
		if IS_AGAIN and do_correction:
			# if again is enabled, means going to do correction, set version to -1
			if self.video_download_timestamp - np.floor(self.video_download_timestamp) >= AGAIN_THRES \
			   and self.video_download_timestamp - np.floor(self.video_download_timestamp) < 1 \
			   and (self.buffer_size_BL >= AGAIN_BL_THRES or self.video_seg_index_BL >= VIDEO_LEN):
			   video_segment_index = int(np.ceil(self.video_download_timestamp))
			   print("Will do correction!")
			   print("Current Time:", self.video_download_timestamp)
			   return current_video_version, video_segment_index
		video_segment_index = -1
		if (self.buffer_size_BL < Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN) or (self.video_seg_index_BL < VIDEO_LEN and \
			 self.buffer_size_BL >= Q_REF_BL and self.buffer_size_EL >= ET_MAX_PRED) :
			current_video_version = 0
			video_segment_index = self.video_seg_index_BL
			# print(self.video_seg_index_BL, self.video_seg_index_EL, int(np.floor(self.video_download_timestamp)))
		elif (self.buffer_size_BL >= Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN and \
			   self.video_seg_index_EL < VIDEO_LEN and self.buffer_size_EL<ET_MAX_PRED) \
			or (self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
			#PI control logic
			u_p = KP * (self.buffer_size_EL - Q_REF_EL)
			u_i = 0
			if len(self.buffer_size_history) != 0:
				# print(self.buffer_size_history)
				# for index in range(0, len(self.buffer_size_history)):
				for index in range(1, min(PI_RANGE+1, len(self.buffer_size_history)+1)):
					# if not IS_REPAIR:
					# 	u_i +=  KI  * (self.buffer_size_history[-index][1] - Q_REF_EL)
					# else:
					u_i +=  KI  * (self.buffer_size_history[-index][1] - Q_REF_EL)
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
			# if R_hat >= rate_cut[3] - rate_cut[0]:
			# 	current_video_version = 3
			# elif R_hat >= rate_cut[2] - rate_cut[0]:
			# 	current_video_version = 2
			# else:
			# 	current_video_version = 1


			if R_hat >= rate_cut[3]:
				current_video_version = 3
			elif R_hat >= rate_cut[2]:
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
		self.already_correct = False
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
		# print(self.network_seg_index)
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
						# if not IS_REPAIR:
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
						# else:
						# 	if len(self.EVR_EL_Recordset) > 0:
						# 		if self.EVR_EL_Recordset[-1][0] < self.video_seg_index_EL:
						# 			# self.buffer_size_EL += 1
						# 			self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
						# 				self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
						# 				self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
						# 				VP_SPAN_YAW, VP_SPAN_PITCH, temprol_eff, RETRAN_UTILI_INIT, 0])
						# 			recording_EL = 1
						# 			# print("Regular channel replace 5G chunk!!!!!!--1-1-1-1")
						# 			# print("Append:", self.video_seg_index_EL)
						# 		elif self.EVR_EL_Recordset[-1][0] == self.video_seg_index_EL:
						# 			if temprol_eff > self.EVR_EL_Recordset[-1][9]+0.2 and self.video_version == MAX_VIDEO_VERSION:
						# 				self.EVR_EL_Recordset[-1][3] = self.yaw_predict_value_quan*30-15
						# 				self.EVR_EL_Recordset[-1][5] = self.pitch_predict_value_quan
						# 				self.EVR_EL_Recordset[-1][9] = temprol_eff
						# 				# self.buffer_size_EL += 1
						# 				recording_EL = 1
						# 				# print("Regular channel replace 5G chunk!!!!!!---")
						# 				# print("update existing:", self.video_seg_index_EL)
						# 			else:
						# 				print("Arrive at same sec, but worse", self.video_seg_index_EL)

						# 		else:
						# 			print("arrive too late", self.video_seg_index_EL)
						# 			print("Shell never been triggered")
						# 			assert(1==0)
						# 			is_same_el_index = -1
						# 		if self.video_seg_index_EL == int(np.floor(self.video_download_timestamp)):
						# 			is_same_el_index = 1
						# 	else:
						# 		self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
						# 			self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
						# 			self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
						# 			VP_SPAN_YAW, VP_SPAN_PITCH, temprol_eff, RETRAN_UTILI_INIT, 0])
						# 		recording_EL = 1

					else:
						is_same_el_index = -1
					self.video_seg_index_EL = max(self.video_seg_index_EL + 1, int(np.ceil(self.video_download_timestamp)))
					# print("el change:", self.video_seg_index_EL)
				else:
					print(self.video_download_timestamp, self.video_seg_index_BL, self.video_seg_index_EL, self.buffer_size_BL, self.buffer_size_EL)
					if self.video_download_timestamp < (np.floor(self.video_download_timestamp) + AGAIN_THRES):
						self.remaining_time = (1-AGAIN_THRES)
						self.downloadedPartialVideo = 0
						self.video_download_timestamp = (np.floor(self.video_download_timestamp) + AGAIN_THRES)
						self.video_segment = 0
						self.video_version = 0


					downlading = self.correct(yaw_trace, pitch_trace, rate_cut, network_trace)
					print(self.video_download_timestamp, temp_video_download_timestamp)
					print('OCOOLDD ')
					if np.ceil(self.video_download_timestamp) == np.ceil(temp_video_download_timestamp):
						# print("Correction!!!!!!!!")
						self.already_correct = True
					else:
						self.already_correct = False
					print(self.video_download_timestamp, self.video_seg_index_BL, self.video_seg_index_EL, self.buffer_size_BL, self.buffer_size_EL)
				
				#### Liyang
				video_rate = rate_cut[self.video_version] #  - rate_cut[0]
				####
			else :
				assert 0 == 1
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

		self.buffer_size_BL = max(0.0, self.buffer_size_BL - (self.video_download_timestamp - temp_video_download_timestamp))
		if not self.downloadedPartialVideo:
			if is_same_el_index  == 0 : ## download one is useful
				if not self.is_empty_el:
					self.buffer_size_EL = max(0.0, self.buffer_size_EL - (self.video_download_timestamp - temp_video_download_timestamp))
			elif is_same_el_index == 1:
				self.buffer_size_EL = max(0.0, self.buffer_size_EL - (self.video_download_timestamp - np.floor(temp_video_download_timestamp)))
			else:
				# print("self buffer length:", self.buffer_size_EL)
				assert(np.round(self.buffer_size_EL == 0))
		else:
			self.buffer_size_EL = max(0.0, self.buffer_size_EL - (self.video_download_timestamp - temp_video_download_timestamp))
		
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
			self.buffer_size_history.append([np.round(self.buffer_size_BL), np.round(self.buffer_size_EL)])
			self.network_seg_index += 1

			# if self.retransmit_available(network_trace_aux) and IS_CORRECT:
			# 	self.correct(yaw_trace, pitch_trace, rate_cut, network_trace_aux)

		return temp_video_download_timestamp, recording_EL


	## Check 5G availability
	# def retransmit_available(self, network_trace_aux):
		
	# 	availability = network_trace_aux[int(np.floor(self.video_download_timestamp))]
	# 	if availability != 0:
	# 		return True
	# 	else:
	# 		return False

	# def repair(self, correct_time, correct_index, yaw_trace, pitch_trace, rate_cut, network_trace_aux):
	# 	if self.correctness_using == 0 and correct_index < 300:	
	# 		# print("Repair:", correct_index)	
	# 		correct_time -= REPAIR_TIME_HEAD
	# 		new_yaw_predict_value, new_yaw_predict_value_quan = self.predict_yaw(yaw_trace, correct_time, correct_index)
	# 		new_pitch_predict_value, new_pitch_predict_value_quan = self.predict_pitch(pitch_trace, correct_time, correct_index)
	# 		## Retransmit the how chunk as previous 

	# 		extra_segment = rate_cut[MAX_VIDEO_VERSION]-rate_cut[0]
	# 		downloading_5G = 0
	# 		new_temprol_eff = 0
	# 		## Start to retransmit
	# 		# retran_bandwidth = np.random.normal(RETRAN_BW_MEAN, RETRAN_BW_STAND)
	# 		retran_bandwidth_1 = network_trace_aux[correct_index-1]
	# 		retran_bandwidth_2 = network_trace_aux[correct_index]
	# 		self.correctness_using = 1
	# 		## Calculate transmitting time, shall be replaced by function with real trace
	# 		if extra_segment > retran_bandwidth_1*REPAIR_TIME_HEAD:
	# 			if retran_bandwidth_2 == 0:
	# 				print("5G Too bad")
	# 				self.correctness_using = 0
	# 				return					
	# 			downloading_5G = (extra_segment - retran_bandwidth_1*REPAIR_TIME_HEAD)/retran_bandwidth_2
	# 			if downloading_5G > REPAIR_TIME_THRES:
	# 				print("5G Too bad")
	# 				self.correctness_using = 0
	# 				self.addition_time += (REPAIR_TIME_THRES + REPAIR_TIME_HEAD)
	# 				return
	# 			new_temprol_eff = min(1, 1 - downloading_5G)
	# 			self.addition_time += (downloading_5G + REPAIR_TIME_HEAD)
	# 		else:
	# 			new_temprol_eff = 1
	# 			self.addition_time += extra_segment/retran_bandwidth_1
	# 		if len(self.EVR_EL_Recordset) > 0:
	# 			assert self.EVR_EL_Recordset[-1][0] < correct_index
	# 		# self.buffer_size_EL += 1

	# 		self.EVR_EL_Recordset.append([correct_index, MAX_VIDEO_VERSION, (retran_bandwidth_1+retran_bandwidth_2)/2, \
	# 			new_yaw_predict_value_quan*30-15, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
	# 			new_pitch_predict_value_quan, pitch_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
	# 			VP_SPAN_YAW, VP_SPAN_PITCH, new_temprol_eff, extra_segment, 1])
	# 		# print("repair degree",new_yaw_predict_value_quan*30-15, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2])
	# 		# print("Effective time", new_temprol_eff)

	# 		self.record_info = np.append(self.record_info, \
	# 				[self.video_download_timestamp, self.buffer_size_BL, self.buffer_size_EL,
	# 				new_yaw_predict_value, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
	# 				new_pitch_predict_value, pitch_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2], correct_time])
	# 		self.correctness_using = 0
	# 	return

	## To use the second 5G high-speed low-latency connection to correct vp
	def correct(self,yaw_trace, pitch_trace, rate_cut, network_trace_aux):
		print("Enter correction")
		if self.correctness_using == 0:
			correct_time = self.video_download_timestamp
			correct_index = self.video_segment_index
			next_EL = []
			el_now = 0

			while self.correct_ptr < len(self.EVR_EL_Recordset) and self.EVR_EL_Recordset[self.correct_ptr][0] < correct_index :
				self.correct_ptr += 1
			if self.correct_ptr >= len(self.EVR_EL_Recordset):
				# if IS_REPAIR and correct_index < VIDEO_LEN:
				# 	# print("Repair:", correct_index)
				# 	# if len(self.EVR_EL_Recordset) > 0:
				# 	# 	print("last one",self.EVR_EL_Recordset[-1][0])
				# 	self.repair(correct_time, correct_index, yaw_trace, pitch_trace, rate_cut, network_trace_aux)
				print("There is no next el video!")
				return False
			else:
				# correct_time -= CORRECT_TIME_HEAD
				if len(self.EVR_EL_Recordset) > 1:
					temp_index = self.EVR_EL_Recordset[self.correct_ptr-1][0]
					if temp_index == correct_index -1:
						el_now = 1
				else:
					el_now = 1
				print("el_now value:", el_now)
				if el_now == 0:
					print(self.EVR_EL_Recordset[-3:])
					print(temp_index, )
				next_EL = self.EVR_EL_Recordset[self.correct_ptr]
				assert (next_EL[9] == 1)
				if next_EL[10] > 0:
					print("Will do second correction!!")
					return False
				# print(next_EL[0])
				if next_EL[0] > correct_index:
					# No need to correct
					## Maybe special case, wrong!! Pay attention
					print("Too far away, not yet.")
					print("Must be handled!")
					return False
				elif next_EL[0] == correct_index:
					print("correct chunk:", correct_index)
					print("correct time:", correct_time)
					direction_central = 0
					if USING_TRUN:
						new_yaw_predict_value, new_yaw_predict_value_quan = self.predict_yaw_trun(yaw_trace, correct_time, next_EL[0])
					else:
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
						print("overlap")
						print("print buffer %f %f" % (self.buffer_size_BL, self.buffer_size_EL))
						return False
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
								# area = [new_left_boundary, new_right_boundary, 1, False]
								area = [right_boundary, new_right_boundary, 1, False]
								print("totally wrong")
						else:
							assert (self.decide_direction(left_boundary, new_left_boundary) == -1)
							# print("right_boundary", left_boundary)
							# print("new_right_boundary", new_left_boundary)
							if self.decide_direction(left_boundary, new_right_boundary == 1):
								area = [new_left_boundary, left_boundary, -1, True]
							else:
								# area = [new_left_boundary, new_right_boundary, -1, False]
								area = [new_left_boundary, left_boundary, -1, False]
								print("totally wrong")
						# print("makeup area", area)
						# print("-----------")
						start_tile, end_tile, total_tiles, tile_left, tile_right, tile_center = self.calculate_tiles(area)
						assert (next_EL[1] >= 1)

						## Calculate size of data should be retransmitted
						retran_data_size = (RETRAN_EXTRA*total_tiles/(VP_SPAN_YAW/TILE_SIZE))*rate_cut[next_EL[1]]   # -rate_cut[0])
						print(retran_data_size)
						print("Before buffer %f %f" % (self.buffer_size_BL, self.buffer_size_EL))
						## Simulate 5G retransmisstion and correct
						downloading_5G = 0
						## Start to retransmit
						retran_bandwidth_1 = network_trace_aux[correct_index-1]
						retran_bandwidth_2 = network_trace_aux[correct_index]
						self.correctness_using = 1
						## Calculate transmitting time, shall be replaced by function with real trace
						
						if retran_data_size > retran_bandwidth_1*CORRECT_TIME_HEAD:
							if retran_bandwidth_2 == 0:
								self.correctness_using = 0
								self.EVR_EL_Recordset[self.correct_ptr][10] = retran_bandwidth_1*CORRECT_TIME_HEAD
								self.addition_time += CORRECT_TIME_HEAD
								self.addition_data += retran_bandwidth_1*CORRECT_TIME_HEAD
								self.video_download_timestamp = np.ceil(self.video_download_timestamp)
								self.downloadedPartialVideo = 0
								self.remaining_time = 1
								self.buffer_size_BL = np.floor(self.buffer_size_BL)
								if el_now == 1:
									self.buffer_size_EL = max(0.0, np.floor(self.buffer_size_EL))
								self.video_seg_index_EL = max(self.video_seg_index_EL, int(np.round(self.video_download_timestamp)))
								print("Second zero")
								print("Return", self.video_download_timestamp)
								self.buffer_size_history.append([np.round(self.buffer_size_BL), np.round(self.buffer_size_EL)])
								self.network_seg_index += 1
								return True
							downloading_5G = (retran_data_size - retran_bandwidth_1*CORRECT_TIME_HEAD)/retran_bandwidth_2
							if downloading_5G >= CORRECT_TIME_THRES:
								self.EVR_EL_Recordset[self.correct_ptr][10] = (retran_bandwidth_1*CORRECT_TIME_HEAD + retran_bandwidth_2*CORRECT_TIME_THRES)
								self.correctness_using = 0
								self.addition_time += (CORRECT_TIME_THRES + CORRECT_TIME_HEAD)
								self.addition_data += (retran_bandwidth_1*CORRECT_TIME_HEAD + retran_bandwidth_2*CORRECT_TIME_THRES)
								self.video_download_timestamp = np.ceil(self.video_download_timestamp) + CORRECT_TIME_THRES
								self.downloadedPartialVideo = 0
								self.remaining_time = 1 - CORRECT_TIME_THRES
								self.buffer_size_BL = np.floor(self.buffer_size_BL) - CORRECT_TIME_THRES
								if el_now == 1:
									self.buffer_size_EL = max(0.0, np.floor(self.buffer_size_EL)-CORRECT_TIME_THRES)
								else:
									self.buffer_size_EL  = max(0.0, self.buffer_size_EL - CORRECT_TIME_THRES)
								print("Second connection too bad!!!")
								print("Download 5G:", downloading_5G)
								print("Bandwidth:", retran_bandwidth_1, retran_bandwidth_2)
								print(self.video_download_timestamp)
								print(self.buffer_size_BL)
								print(self.buffer_size_EL)
								self.buffer_size_history.append([np.ceil(self.buffer_size_BL), np.ceil(self.buffer_size_EL)])
								self.network_seg_index += 1
								self.video_seg_index_EL = max(self.video_seg_index_EL, int(np.ceil(self.video_download_timestamp)))
								return True
							else:
								new_temprol_eff = 1 - downloading_5G
								self.addition_time += (downloading_5G + CORRECT_TIME_HEAD)
								self.addition_data += retran_data_size
								self.addition_useful_data += retran_data_size
								self.video_download_timestamp = np.ceil(self.video_download_timestamp) + downloading_5G
								self.buffer_size_BL = np.floor(self.buffer_size_BL)-downloading_5G
								if el_now == 1:
									self.buffer_size_EL = max(0.0, np.floor(self.buffer_size_EL)-downloading_5G)
								else:
									self.buffer_size_EL = max(0.0, self.buffer_size_EL - downloading_5G)
								self.video_seg_index_EL = max(self.video_seg_index_EL, int(np.ceil(self.video_download_timestamp)))
								self.buffer_size_history.append([np.round(self.buffer_size_BL), np.round(self.buffer_size_EL)])
								self.network_seg_index += 1
						else:
							new_temprol_eff = 1
							self.addition_time += (retran_data_size/retran_bandwidth_1)
							self.addition_data += retran_data_size
							self.addition_useful_data += retran_data_size
							self.video_download_timestamp += retran_data_size/retran_bandwidth_1
							self.buffer_size_BL -= retran_data_size/retran_bandwidth_1
							if el_now == 1:
								self.buffer_size_EL = max(0.0, self.buffer_size_EL - retran_data_size/retran_bandwidth_1)
							self.video_seg_index_EL = max(self.video_seg_index_EL, int(np.ceil(self.video_download_timestamp)))
							# self.buffer_size_history.append([self.buffer_size_BL, self.buffer_size_EL])
							# self.network_seg_index += 1
						print("after buffer %f %f" % (self.buffer_size_BL, self.buffer_size_EL))
						print(self.video_download_timestamp)
						print("will update!!!!!!!!!!!!!")
						new_yaw_span = 0.0
						new_central = 0.0
						## If continueous
						# if area[3]:
						# new_distance = min(np.abs(area[0] - area[1]), 360 - np.abs(area[0] - area[1]))
						# new_yaw_span = VP_SPAN_YAW + new_distance
						if area[2] == 1:
							right = tile_right
							left = left_boundary
						else:
							right = right_boundary
							left = tile_left
						if left <= right:
							new_central = (right+left)/2.0
							new_yaw_span = right-left
						else:
							new_central = (right+left+360.0)/2.0
							new_yaw_span = 360.0-left+right
						if new_central >= 360.0:
							new_central -= 360.0
						if new_central < 0:
							new_central += 360.0
						##udpate EL recordest
						self.EVR_EL_Recordset[self.correct_ptr][3] = new_central
						self.EVR_EL_Recordset[self.correct_ptr][7] = new_yaw_span
						#######
						# if distance >= 60.0 and new_temprol_eff <= 0.1:
						# if new_temprol_eff <= 0.1:
						self.EVR_EL_Recordset[self.correct_ptr][9] = new_temprol_eff
						#######
						self.EVR_EL_Recordset[self.correct_ptr][10] = retran_data_size
						self.correctness_using = 0
						print("Update EL", new_central, new_yaw_span, new_temprol_eff, retran_data_size, self.EVR_EL_Recordset[self.correct_ptr][10])
						print("area:", distance, area)
						# else:
						# 	## Liyang, should modify to update EL recordest
						# 	return
							## 5G is not usable
						# print("Could correct!")
						# print("previous central", central)
						# print("new central", new_yaw_predict_value)
						# print("direction", direction_central)
						# print("------------------------------")
		else:
			## 5G is being used
			print("Correct function is not available right now.")

		return True


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
		total_tiles = 0.0
		first_tile = float(int(area[0]/TILE_SIZE))
		last_tile = 0.0
		if area[1] == 0:
			last_tile = 0
		else:
			last_tile = int(area[1]/TILE_SIZE)
		new_left = first_tile*TILE_SIZE
		new_right = last_tile*TILE_SIZE + TILE_SIZE - 1 
		new_c = 0.0
		if first_tile <= last_tile:
			total_tiles = last_tile - first_tile + 1
			new_c = (new_left+new_right)/2.0
		else:
			total_tiles = MAX_TILES - first_tile + last_tile + 1
			new_c = (new_left + new_right + 360)/2.0
			if new_c >= 360:
				new_c -= 360
		# print(area, total_tiles)
		assert first_tile < MAX_TILES
		assert last_tile < MAX_TILES

		return first_tile, last_tile, total_tiles, new_left, new_right, new_c

	def predict_yaw(self, yaw_trace, video_download_timestamp, video_segment_index):
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

	def predict_yaw_trun(self, yaw_trace, video_download_timestamp, video_segment_index):
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
			new_value = [vp_value[-1], vp_value[-2]]
			new_index = [vp_index[-1], vp_index[-2]]
			sign = np.sign(vp_value[-1] - vp_value[-2])
			temp = vp_value[-2]
			sign_index = -3
			# while sign == 0.0 and sign_index >= -VIEW_PRED_SAMPLE_LEN:
			# 	sign = np.sign(new_value[-1] - vp_value[sign_index])
			# 	new_value.append(vp_value[sign_index])
			# 	new_index.append(vp_index[sign_index])
			# 	temp = vp_value[sign_index]	
			# 	sign_index -= 1
			# assert (sign != 0)
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
	# network_trace = loadNetworkTrace(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	half_sec_network_trace, network_trace = load_5G.load_5G_Data(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	#### For 5G delay
	network_delay = load_5G.load_5G_latency(DELAY_TRACE)
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
	# print(min(network_trace))
	network_trace_aux = network_trace
	# if CELL == 0: ## 4G trace
	# 	network_trace_aux = loadNetworkTrace(NETWORK_TRACE_FILENAME, EXTRA_MULTIPLE, EXTRA_ADD)
	# elif CELL == 1: ## 5G trace
	# 	network_trace_aux = load_5G.load_5G_Data(NETWORK_TRACE_FILENAME, EXTRA_MULTIPLE, EXTRA_ADD)
	# else:
	# 	print("Load BW trace error!!")
	# print(min(network_trace_aux))
	# yaw_trace, pitch_trace = loadViewportTrace()
	yaw_trace, pitch_trace = loadViewportTrace_new()
	network_pdf, pdf_bins, pdf_patches = plot_pdf(network_trace)
	network_cdf, cdf_bins, cdf_patches = plot_cdf(network_trace)
	# extra_network_pdf, extra_pdf_bins, extra_pdf_patches = plot_pdf(network_trace_aux)
	# extra_network_cdf, extra_cdf_bins, extra_cdf_patches = plot_cdf(network_trace_aux)
	rate_cut = new_rate_determine(network_cdf)	## Final optimum value
	# rate_cut = rate_determine(network_cdf)		## Percentile method
	# rate_cut = gamma_rate_determine(network_trace)  ## Genrate gamma, very low rate
	video_trace = loadVideoTrace(rate_cut)
	streaming_sim = streaming()
	streaming_sim.run(network_trace, yaw_trace, pitch_trace, video_trace, rate_cut, network_trace_aux)
	# print(len(streaming_sim.buffer_size_history))
	if IS_DEBUGGING:
		display(streaming_sim.record_info, streaming_sim.EVR_BL_Recordset, streaming_sim.EVR_EL_Recordset, \
			rate_cut, yaw_trace, pitch_trace, network_trace, streaming_sim.bw_info, streaming_sim.buffer_size_history, network_trace_aux, half_sec_network_trace)
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

def loadViewportTrace_new():
	mat_contents = sio.loadmat(VIEWPORT_TRACE_FILENAME_NEW)
	trace_data = mat_contents['data_alpha_beta'] # array of structures
	# pitch_trace_data = mat_contents['view_angle_pitch_combo'] + 90 # array of structures
	# print(len(yaw_trace_data[0]), len(pitch_trace_data[0]))
	print(trace_data.T.shape)

	yaw_trace_data = (trace_data.T[1]/math.pi)*180+180
	pitch_trace_data = (trace_data.T[2]/math.pi)*180
	# print(yaw_trace_data.shape, yaw_trace_data[:VIDEO_LEN*VIDEO_FPS])
	yaw_trace_data =  [x for x in yaw_trace_data if not math.isnan(x)]
	pitch_trace_data =  [x for x in pitch_trace_data if not math.isnan(x)]
	# yaw_trace_data.tolist().remove(float('nan'))
	# pitch_trace_data.tolist().remove(float('nan'))
	# print(np.array(yaw_trace_data).shape)
	# assert (len(yaw_trace_data[0]) > VIDEO_LEN*VIDEO_FPS and len(pitch_trace_data[0]) > VIDEO_LEN*VIDEO_FPS)
	return yaw_trace_data[:VIDEO_LEN*VIDEO_FPS], pitch_trace_data[:VIDEO_LEN*VIDEO_FPS]
def loadVideoTrace(rate_cut):
	video_trace0 = rate_cut[0] * np.ones(VIDEO_LEN)
	# video_trace1 = (rate_cut[1] - rate_cut[0]) * np.ones(VIDEO_LEN)
	# video_trace2 = (rate_cut[2] - rate_cut[0]) * np.ones(VIDEO_LEN)
	# video_trace3 = (rate_cut[3] - rate_cut[0]) * np.ones(VIDEO_LEN)
	video_trace1 = rate_cut[1] * np.ones(VIDEO_LEN)
	video_trace2 = rate_cut[2] * np.ones(VIDEO_LEN)
	video_trace3 = rate_cut[3] * np.ones(VIDEO_LEN)
	return [video_trace0, video_trace1, video_trace2, video_trace3]

def plot_pdf(trace):
	global FIGURE_NUM
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	n, bins, patches = plt.hist(trace, range(0, int(np.ceil(max(trace))) + 1), normed = 1, label='PDF', color='b')
	plt.title('5G_Trace_1 Bandwidth PDF')
	plt.xlabel('Mbps', fontsize=15)
	# plt.ylabel('', fontsize=15)
	plt.tick_params(axis='both', which='major', labelsize=15)
	# plt.tick_params(axis='both', which='minor', labelsize=15)
	plt.axis([0, max(trace), 0, max(n)])
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def plot_cdf(trace):
	global FIGURE_NUM
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM +=1
	n, bins, patches = plt.hist(trace, range(0,int(np.ceil(max(trace))) + 1), normed = 1, cumulative=True, label='CDF', histtype='stepfilled', color='b')
	plt.title('5G_Trace_1 Bandwidth CDF')
	plt.xlabel('Mbps', fontsize=15)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.axis([0, max(trace), 0, 1.2])
	# print(a,b,c)
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def rate_determine(network_cdf):
	# cut_percent = [0.2, 0.5, 0.8]
	# rate_cut = [10.0]
	

	cut_percent = [0.2, 0.4, 0.6, 0.8]
	rate_cut = []

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

def new_rate_determine(network_cdf):
	# cut_percent = [0.4, 0.6, 0.8]
	# rate_cut = [10.0]
	# cut_idx = 0
	# for cum in network_cdf:
	# 	if cum >= np.max(network_cdf) * cut_percent[cut_idx]:
	# 		rate_cut.append(np.round(network_cdf.tolist().index(cum)))
	# 		cut_idx += 1
	# 		if cut_idx >= len(cut_percent): 
	# 			break
	rate_cut = [0.0]*BITRATE_LEN
	
	rate_list = [11,12,21,22,31,32]
	assert VER in rate_list
	if VER == 11:
		##  3 9
		rate_cut[0] = 50.6
		rate_cut[1] = 512.8
		rate_cut[2] = 683.74
		rate_cut[3] = 854.68
	elif VER == 12:
		##  3 13
		rate_cut[0] = 147.95
		rate_cut[1] = 439.79
		rate_cut[2] = 586.39
		rate_cut[3] = 732.99
	elif VER == 21:
		##  1 9
		rate_cut[0] = 55.02
		rate_cut[1] = 453.49
		rate_cut[2] = 604.65
		rate_cut[3] = 755.81
	elif VER == 22:
		##  1 13
		rate_cut[0] = 152.23
		rate_cut[1] = 380.58
		rate_cut[2] = 507.44
		rate_cut[3] = 634.3
	elif VER == 31:
		##  2 9
		rate_cut[0] = 104.37
		rate_cut[1] = 360.71
		rate_cut[2] = 480.94
		rate_cut[3] = 601.18
	elif VER == 32:

		##  2 13
		rate_cut[0] = 183.5
		rate_cut[1] = 301.35
		rate_cut[2] = 401.8
		rate_cut[3] = 502.25
	else:
		assert 0 == 1


	if IS_DEBUGGING: 
		print('Rate0 = %f, Rate1 = %f, Rate2 = %f, Rate3 = %f' % (rate_cut[0],rate_cut[1],rate_cut[2],rate_cut[3]))
	return rate_cut

def gamma_rate_determine(network_trace):
	# cut_percent = [0.4, 0.6, 0.8]
	# rate_cut = [10.0]
	# cut_idx = 0
	# for cum in network_cdf:
	# 	if cum >= np.max(network_cdf) * cut_percent[cut_idx]:
	# 		rate_cut.append(np.round(network_cdf.tolist().index(cum)))
	# 		cut_idx += 1
	# 		if cut_idx >= len(cut_percent): 
	# 			break
	average = np.mean(network_trace)
	rate0 = average* AVE_RATIO * 0.3
	rate2 = average* AVE_RATIO - rate0
	rate1 = rate2*0.75
	rate3 = rate2*1.25
	rate_cut = [rate0, rate1, rate2, rate3]
	if IS_DEBUGGING: 
		print('Rate0 = %f, Rate1 = %f, Rate2 = %f, Rate3 = %f' % (rate_cut[0],rate_cut[1],rate_cut[2],rate_cut[3]))
	return rate_cut

def display(record_info, EVR_BL_Recordset, EVR_EL_Recordset, rate_cut, yaw_trace, pitch_trace, network_trace, bw_info, buffer_size_history, network_trace_aux, half_sec_network_trace):
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
	plt.axis([0, 300, 0, 15])

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
	plt.axis([0, VIDEO_LEN, 0, 450])

	p = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	plt.plot(range(1,2*(VIDEO_LEN),2), network_trace,'r-', label='Real-time Bandwidth', linewidth=1.2)
	plt.plot(range(1,2*(VIDEO_LEN+1)), half_sec_network_trace,'b-', label='Previous Bandwidth', linewidth=1.2)
	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right')
	# plt.title('Bandwidth Predict and Real Trace')
	plt.title('5G Bandwidth Trace')
	plt.xlabel('Second', fontsize=15)
	plt.ylabel('Mbps', fontsize=15)
	plt.tick_params(axis='both', which='major', labelsize=15)
	plt.tick_params(axis='both', which='minor', labelsize=15)
	plt.axis([0, NETWORK_TRACE_LEN, 0, max(network_trace)+200])
	# plt.axis([0, 300, 0, max(network_trace_aux)+50])


	display_bitrate = [0.0]*VIDEO_LEN
	receive_bitrate = [0.0]*VIDEO_LEN
	log_bitrate = [0.0]*VIDEO_LEN
	extra_cost = [[0.0, 0]]*VIDEO_LEN
	extra_repair = [[0.0, 0]]*VIDEO_LEN
	el_coverage_ratio = 0.0

	r_base = R_MIN
	# r_base = rate_cut[0]/6

	# print(EVR_BL_Recordset)
	for i in range (0,BUFFER_BL_INIT):
		display_bitrate[i] += (rate_cut[0]/6)
		receive_bitrate[i] += (rate_cut[0]/6)
	# display_bitrate[0] += (rate_cut[3]-rate_cut[0]/6)
	# receive_bitrate[0] += (rate_cut[3]-rate_cut[0]/6)
	display_bitrate[0] += rate_cut[3]
	receive_bitrate[0] += rate_cut[3]
	
	# log_bitrate[0] += math.log10((rate_cut[3]-rate_cut[0])/r_base)
	for i in range(0, len(EVR_BL_Recordset)):
		display_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]/6
		receive_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]/6
	total_correction = 0.0
	total_repair = 0.0
	total_gamma = 0.0
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
		el_coverage_ratio += sum_eff
		total_gamma += EVR_EL_Recordset[i][9]
		display_bitrate[EVR_EL_Recordset[i][0]] += EVR_EL_Recordset[i][9]*sum_eff*rate_cut[EVR_EL_Recordset[i][1]]    ##  -rate_cut[0]/6) ##+ (1-EVR_EL_Recordset[i][9])*rate_cut[0]/6
		receive_bitrate[EVR_EL_Recordset[i][0]] += rate_cut[EVR_EL_Recordset[i][1]]  ##    -rate_cut[0]/6)
		# if (EVR_EL_Recordset[i][9]*sum_eff*(rate_cut[EVR_EL_Recordset[i][1]]-rate_cut[0]/6)) != 0:
		# 	log_bitrate[EVR_EL_Recordset[i][0]] += math.log10((EVR_EL_Recordset[i][9]*sum_eff*(rate_cut[EVR_EL_Recordset[i][1]]-rate_cut[0]/6))/r_base)

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
	for i in range(len(display_bitrate)):
		# assert receive_bitrate[i] >= r_base
		log_bitrate[i] += math.log10(display_bitrate[i]/r_base)
	extra_cost = np.array(extra_cost).T
	extra_repair = np.array(extra_repair).T
	# total_correction = np.sum(extra_cost)
	# total_repair = np.sum(extra_repair)
	print("Total cost data: %f" % (total_correction))
	print("Total repair data: %f" % (total_repair))
	print("Total extra data: %f" % (total_repair+total_correction))
	print("Average EL coverage ratio:", (el_coverage_ratio+BUFFER_EL_INIT)/VIDEO_LEN)
	print("Average Effective coverage ratio:", (el_coverage_ratio+BUFFER_EL_INIT)/len(EVR_EL_Recordset))
	print("EL existing ratio:", float(len(EVR_EL_Recordset)+BUFFER_EL_INIT)/VIDEO_LEN)

		# print (sum_eff, EVR_EL_Recordset[i][0])
		# print(EVR_EL_Recordset[i][3]*30-15, EVR_EL_Recordset[i][3], EVR_EL_Recordset[i][5],EVR_EL_Recordset[i][9])
	# return
	print("Effective bitrate:", sum(display_bitrate))
	print("Received bitrate:", sum(receive_bitrate))
	print("Log bitrate:", sum(log_bitrate))
	print("Gamma value:", (total_gamma+BUFFER_EL_INIT)/VIDEO_LEN )



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
	plt.axis([0, 300, 0, max(receive_bitrate) + 100])

	if IS_AGAIN:
		q = plt.figure(FIGURE_NUM)
		FIGURE_NUM += 1
		plt.bar(extra_cost[1], extra_cost[0], color='red', label='Correction Data', edgecolor = "red")
		plt.bar(extra_repair[1], extra_repair[0], color='blue', label='Retransmission Data', edgecolor = "blue")
		# plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-', label='Received Video Bitrate')
		plt.legend(loc='best')
		plt.title('Extra Data Cost')
		plt.xlabel('Second')
		plt.ylabel('MB')
		plt.axis([0, 300, 0, max(max(extra_cost[0])+1, max(extra_repair[0])+1)])

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
	if IS_AGAIN:
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
	
