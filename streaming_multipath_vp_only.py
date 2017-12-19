
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G 
# Display buffer initialization
# BUFFER_BL_INIT = 10
BUFFER_EL_INIT = 1
# Q_REF_BL = 10
Q_REF_EL = 1
ET_MAX_PRED = Q_REF_EL + 1

# Control parameters
IS_SAVING = 0
IS_DEBUGGING = 1
DISPLAYRESULT = 1
TRACE_IDX = 1
KP = 0.8		# P controller
KI = 0.01		# I controller
PI_RANGE = 60

# Video parameters
VIDEO_LEN = 600		# seconds
VIDEO_FPS = 30		# hz	
NETWORK_TRACE_LEN = VIDEO_LEN + 100 		# seconds
VIEW_PRED_SAMPLE_LEN = 30	# samples used for prediction
POLY_ORDER = 1				#  1: linear   2: quadratic
FRAME_MV_LIMIT = 180		# horizontal motion upper bound in degree, used for horizontal circular rotation
KB_IN_MB = 1000  			# actually not
FIGURE_NUM = 1
#
INIT_BW = 30
BW_PRED = 2
BW_PRED_SAMPLE_SIZE = 3
MAX_VIDEO_VERSION = 2
# RTT
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
TILE_SIZE = 22.5
MAX_TILES = int(360.0/TILE_SIZE)
# Directory
REGULAR_CHANNEL_TRACE = 'BW_Trace_4.mat'
REGULAR_MULTIPLE = 50
REGULAR_ADD = 10

VERSION = 1
if VERSION == 1:
	NETWORK_TRACE_FILENAME = 'BW_Trace_5G_2.txt'
	EXTRA_MULTIPLE = 1
	EXTRA_ADD = 0
	CORRECT_TIME_HEAD = 0.01
	REPAIR_TIME_HEAD = 0.01
	IS_CORRECT = 1
	IS_REPAIR = 0
	CORRECT_TIME_THRES = 0.2
	REPAIR_TIME_THRES = 1

elif VERSION == 2:
	NETWORK_TRACE_FILENAME = 'BW_Trace_5G_2.txt'
	EXTRA_MULTIPLE = 1
	EXTRA_ADD = 0
	CORRECT_TIME_HEAD = 0.01
	REPAIR_TIME_HEAD = 0.01
	IS_CORRECT = 1
	IS_REPAIR = 1
	CORRECT_TIME_THRES = 0.2
	REPAIR_TIME_THRES = 1

else:
	NETWORK_TRACE_FILENAME = 'BW_Trace_5G_2.txt'
	EXTRA_MULTIPLE = 1
	EXTRA_ADD = 0
	CORRECT_TIME_HEAD = 0.01
	REPAIR_TIME_HEAD = 0.01
	IS_CORRECT = 0
	IS_REPAIR = 0
	CORRECT_TIME_THRES = 0.2
	REPAIR_TIME_THRES = 1


VIEWPORT_TRACE_FILENAME1 = 'view_angle_combo_video1.mat'
# VIEWPORT_TRACE_FILENAME2 = 'view_trace_fanyi_amsterdam_2D.mat'

class streaming(object):

	def __init__(self):
		# self.video_seg_index_BL = BUFFER_BL_INIT 
		self.video_seg_index_EL = BUFFER_EL_INIT
		self.network_seg_index = 0
		self.remaining_time = 1
		self.video_download_timestamp = 0
		# self.buffer_size_BL = BUFFER_BL_INIT
		self.buffer_size_EL = BUFFER_EL_INIT
		self.buffer_size_history = []
		self.downloadedPartialVideo = 0
		self.correct_ptr = 0
		self.delay = DELAY
		self.correctness_using = 0
		self.request_pre = 0
		self.do_correct = IS_CORRECT
		self.freezing = 0
		self.freezing_time_current_chunk = 0.0
		self.total_freezing = 0.0
		self.display_time = 0.0
		self.remaining_buffer = BUFFER_EL_INIT
		self.less = 1
		self.addition_time = 0.0

		# self.EVR_BL_Recordset = []
		self.EVR_EL_Recordset = []
		self.video_segment = 0
		# self.GBUFFER_BL_EMPTY_COUNT = 0
		self.video_version = 0
		# self.video_segment_index = 0
		self.yaw_predict_value = 0
		self.pitch_predict_value = 0
		self.yaw_predict_value_quan = 0
		self.pitch_predict_value_quan = 0
		self.record_info = []
		self.bw_info = []
		self.video_bw_history = []



	def run(self, network_trace, yaw_trace, pitch_trace, video_trace, rate_cut, network_trace_aux):
		# while self.video_seg_index_BL < VIDEO_LEN or \
		# 	(self.video_seg_index_BL >= VIDEO_LEN and self.video_seg_index_EL < VIDEO_LEN):
		while self.video_seg_index_EL < VIDEO_LEN :
			if not self.downloadedPartialVideo:
				if self.buffer_size_EL > ET_MAX_PRED:
					self.less = 0
				else:
					self.less = 1
				if BW_PRED == 1:
					sniff_BW = self.getCurrentBW(network_trace)
				elif BW_PRED == 2:
					sniff_BW = self.predictBW()

				self.bw_info = np.append(self.bw_info, [sniff_BW, self.video_download_timestamp])

				self.video_version = self.control(rate_cut, sniff_BW)
				self.video_segment = video_trace[self.video_version][self.video_seg_index_EL]
				temp_index = self.video_seg_index_EL
				self.yaw_predict_value, self.yaw_predict_value_quan = self.predict_yaw(yaw_trace, self.display_time, temp_index)
				self.pitch_predict_value, self.pitch_predict_value_quan = self.predict_pitch(pitch_trace, self.display_time, temp_index)
			repair = 0
			previous_time, repair = self.video_fetching(network_trace, rate_cut, yaw_trace, pitch_trace, network_trace_aux)
			# return
			## SHOULD MODIFY
			# if not self.downloadedPartialVideo and not repair:
			if not repair:
			# 	self.record_info = np.append(self.record_info, \
			# 		[self.video_download_timestamp, self.buffer_size_BL, self.buffer_size_EL,
			# 		temp_yaw, yaw_trace[int((self.video_seg_index_EL - 1)*VIDEO_FPS-VIDEO_FPS/2)],
			# 		temp_pitch, pitch_trace[int((self.video_seg_index_EL - 1)*VIDEO_FPS-VIDEO_FPS/2)],
			# 		sniff_BW, previous_time])
				self.record_info = np.append(self.record_info, \
					[self.video_download_timestamp, self.buffer_size_EL,
					self.yaw_predict_value, yaw_trace[int(temp_index*VIDEO_FPS+VIDEO_FPS/2)],
					self.pitch_predict_value, pitch_trace[int(temp_index*VIDEO_FPS+VIDEO_FPS/2)], previous_time])
		# print(self.video_bw_history)
		print(self.video_seg_index_EL)
		print('Simluation done')
		print(len(self.EVR_EL_Recordset))
		print(self.video_download_timestamp)
		print("Total freezing:", self.total_freezing)
		print(self.display_time)
		print(self.addition_time)
		# print(self.GBUFFER_BL_EMPTY_COUNT)
		# print(len(self.EVR_BL_Recordset))	

	def control(self, rate_cut, sniff_BW):
		# print(self.buffer_size_BL)
		# print(self.video_seg_index_BL)
		# print(self.downloadedPartialVideo)
		# if not self.downloadedPartialVideo:
		current_video_version = -1
		# video_segment_index = -1
		# if self.buffer_size_BL < Q_REF_BL and self.video_seg_index_BL < VIDEO_LEN :
		# 	current_video_version = 0
		# 	video_segment_index = self.video_seg_index_BL
			# print(self.video_seg_index_BL, self.video_seg_index_EL, int(np.floor(self.video_download_timestamp)))
		if self.video_seg_index_EL < VIDEO_LEN:
			#PI control logic
			u_p = KP * (self.buffer_size_EL - Q_REF_EL)
			u_i = 0
			if len(self.buffer_size_history) != 0:
				# print(self.buffer_size_history)
				# for index in range(0, len(self.buffer_size_history)):
				for index in range(1, min(PI_RANGE+1, len(self.buffer_size_history)+1)):
					u_i +=  KI  * (self.buffer_size_history[-index] - Q_REF_EL)
			
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
			if R_hat >= rate_cut[2]:
				current_video_version = 2
			elif R_hat >= rate_cut[1]:
				current_video_version = 1
			else:
				current_video_version = 0

			# if len(self.video_bw_history) != 0:
			# 	self.video_version = min(current_video_version, self.video_bw_history[-1][2] + 1)
			# else: 
			# self.video_version = current_video_version
			# video_segment_index = self.video_seg_index_EL
			# print('time:', self.video_download_timestamp)

		return current_video_version

	def video_fetching(self, network_trace, rate_cut, yaw_trace, pitch_trace, network_trace_aux):
		repair = 0
		temp_video_download_timestamp = self.video_download_timestamp
		temp_display_time = self.display_time
		# video_rate = 0  #### Liyang
		# self.remaining_buffer = self.buffer_size_EL - np.floor(self.buffer_size_EL)

		if not self.less:
			print("buffer size too long, buffer: %f, time: %f, should wait" % (self.buffer_size_EL, self.video_download_timestamp))
			print("buffer size too long, remaining_buffer: %f, remaining_time: %f" % (self.remaining_buffer, self.remaining_time))
			# if self.remaining_buffer <= self.remaining_time:
			self.display_time += self.remaining_buffer
			self.video_download_timestamp += self.remaining_buffer
			if self.video_download_timestamp == np.floor(self.video_download_timestamp):
				self.remaining_time = 1
			else:
				self.remaining_time = np.ceil(self.video_download_timestamp) - self.video_download_timestamp
			self.buffer_size_EL -= self.remaining_buffer
			self.remaining_buffer = 1
			self.downloadedPartialVideo = 0
			self.video_segment = 0
			if IS_CORRECT and self.retransmit_available(network_trace_aux, self.video_download_timestamp - CORRECT_TIME_HEAD):
				print("We are going to correct:", int(np.round(self.display_time)), self.display_time)
				repair = self.correct(yaw_trace, pitch_trace, rate_cut, network_trace_aux)
			# else:
			# 	self.downloadedPartialVideo = 0
			# 	self.video_download_timestamp += self.remaining_time
			# 	self.video_segment = 0
			# 	self.buffer_size_EL -= self.remaining_time
			# 	self.remaining_buffer -= self.remaining_time
			# 	self.remaining_time = 1


		## Buffer is less than threshold
		else:
			if not self.downloadedPartialVideo and self.delay:
				## If RTT_1 time is larger than remaining times
				self.request_pre = max(0, RTT_1-self.remaining_time)
				self.video_download_timestamp = min(np.ceil(temp_video_download_timestamp), self.video_download_timestamp + RTT_1)
				self.remaining_time = max(0, self.remaining_time - RTT_1)
				if self.freezing:
					print("Something wrong, this should be not triggered!!!")
					self.freezing_time_current_chunk += RTT_1

			if not self.freezing:
				print("NOT freezing(START), buffer size: %f, current time: %f" % (self.buffer_size_EL, self.video_download_timestamp))
				print("NOT freezing(START), remaing buffer: %f, remaing time: %f" % (self.remaining_buffer, self.remaining_time))
				assert(self.buffer_size_EL > 0)
				# print(self.video_segment, network_trace[self.network_seg_index]*self.remaining_time)

				## Remaining buffer time is greater than remaining time
				if self.remaining_buffer > self.remaining_time:

					## If remaining time is enough to download current chunk
					if network_trace[self.network_seg_index]*self.remaining_time >= self.video_segment:
					# if self.video_version == 0:
					# 	self.EVR_BL_Recordset.append([self.video_seg_index_BL, self.video_version, network_trace[self.network_seg_index]])
					# 	self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
					# 	self.remaining_time -= (self.video_download_timestamp - temp_video_download_timestamp)
					# 	# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
					# 	self.downloadedPartialVideo = 0
					# 	self.video_seg_index_BL += 1
					# 	if self.video_seg_index_BL > int(np.floor(self.video_download_timestamp)):
					# 		self.buffer_size_BL += 1
					# 	elif self.video_seg_index_BL == int(np.floor(self.video_download_timestamp)):
					# 		self.buffer_size_BL += (np.ceil(self.video_download_timestamp) - self.video_download_timestamp)
					# 	video_rate = rate_cut[0] ### Liyang
						if self.video_version >= 0:
							self.display_time += self.video_segment/(network_trace[self.network_seg_index])
							self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
							# if self.freezing:
							# 	self.freezing_time_current_chunk += self.video_segment/(network_trace[self.network_seg_index])
							self.remaining_time -= self.video_segment/(network_trace[self.network_seg_index])
							# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
							self.remaining_buffer -= self.video_segment/(network_trace[self.network_seg_index])
							if self.remaining_buffer == 0:
								self.remaining_buffer = 1
							self.buffer_size_EL -= self.video_segment/(network_trace[self.network_seg_index])
							self.buffer_size_EL += 1
							self.downloadedPartialVideo = 0
							# temprol_eff = 1 - min(1, max(self.video_download_timestamp - self.video_seg_index_EL, 0))
							# if self.video_seg_index_EL > int(np.floor(self.video_download_timestamp)):
							# 	self.buffer_size_EL += 1
							# elif self.video_seg_index_EL == int(np.floor(self.video_download_timestamp)):
							# 	self.buffer_size_EL += (np.ceil(self.video_download_timestamp) - self.video_download_timestamp)
							self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
								self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
								self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
								VP_SPAN_YAW, VP_SPAN_PITCH, RETRAN_UTILI_INIT, self.freezing_time_current_chunk])
							# self.video_seg_index_EL = max(self.video_seg_index_EL + 1, int(np.ceil(self.video_download_timestamp)))
							self.video_seg_index_EL += 1						
							#### Liyang
							# video_rate = rate_cut[self.video_version] - rate_cut[0]
							####
							print("NOT FREEZING, time < buffer, complete")
						else :
							print("Unknown video version.")
							# self.display_time = np.ceil(self.display_time + (10.**-8))
							# self.remaining_time = 1
							# self.downloadedPartialVideo = 0
							# self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
							# self.remaining_buffer -= (self.video_download_timestamp - temp_video_download_timestamp)
							# self.video_segment = 0
					else:
						# if self.video_version == 0:
						# 	print('Download base tier, bandwidth is not enough.')
						# elif self.video_version >= 1:
						# 	print('Download enhancement tier, bandwidth is not enough.')
						self.display_time += self.remaining_time
						self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
						self.video_segment = self.video_segment - network_trace[self.network_seg_index]*self.remaining_time
						self.downloadedPartialVideo = 1
						self.request_pre = 0
						print("buffer %f" % self.buffer_size_EL )
						print("remaing time %f" % self.remaining_time)
						self.buffer_size_EL = max(0, self.buffer_size_EL - self.remaining_time)
						self.remaining_buffer = self.remaining_buffer - self.remaining_time
						if self.remaining_buffer == 0 and self.buffer_size_EL >= 1:
							self.remaining_buffer = 1
						if self.buffer_size_EL == 0:
							self.freezing = 1
						self.remaining_time = 1
						print("NOT freezing, time < buffer, uncomplete")
						# self.network_seg_index += 1

				### In this case, remaining buffer time is less than remaining time
				else:
					if network_trace[self.network_seg_index]*self.remaining_buffer >= self.video_segment:
						if self.video_version >= 0:
							self.display_time += self.video_segment/(network_trace[self.network_seg_index])
							self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
							# if self.freezing:
							# 	self.freezing_time_current_chunk += self.video_segment/(network_trace[self.network_seg_index])
							self.remaining_time -= self.video_segment/(network_trace[self.network_seg_index])
							# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
							self.remaining_buffer -= self.video_segment/(network_trace[self.network_seg_index])
							self.buffer_size_EL -= self.video_segment/(network_trace[self.network_seg_index])
							self.downloadedPartialVideo = 0
							self.buffer_size_EL += 1
							if self.remaining_buffer == 0:
								self.remaining_buffer = 1
							# temprol_eff = 1 - min(1, max(self.video_download_timestamp - self.video_seg_index_EL, 0))
							# if self.video_seg_index_EL > int(np.floor(self.video_download_timestamp)):
							# 	self.buffer_size_EL += 1
							# elif self.video_seg_index_EL == int(np.floor(self.video_download_timestamp)):
							# 	self.buffer_size_EL += (np.ceil(self.video_download_timestamp) - self.video_download_timestamp)
							self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
								self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
								self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
								VP_SPAN_YAW, VP_SPAN_PITCH, RETRAN_UTILI_INIT, self.freezing_time_current_chunk])
							# self.video_seg_index_EL = max(self.video_seg_index_EL + 1, int(np.ceil(self.video_download_timestamp)))
							self.video_seg_index_EL += 1						
							#### Liyang
							# video_rate = rate_cut[self.video_version] - rate_cut[0]
							####
						else :
							print("Unknown video version.")
							# self.display_time += self.remaining_buffer
							# self.remaining_time -= self.remaining_buffer
							# self.downloadedPartialVideo = 0
							# self.video_download_timestamp += self.remaining_buffer
							# self.buffer_size_EL -= self.remaining_buffer
							# if self.buffer_size_EL > 0:
							# 	self.remaining_buffer = 1
							# self.video_segment = 0
						print("NOT FREEZING, time > buffer, complete", self.display_time)
					## Remaining buffer is less than remaining time, and not enough
					else:
						# if self.video_version == 0:
						# 	print('Download base tier, bandwidth is not enough.')
						# elif self.video_version >= 1:
						# 	print('Download enhancement tier, bandwidth is not enough.')
						self.display_time += self.remaining_buffer
						self.video_download_timestamp += self.remaining_buffer
						self.video_segment = self.video_segment - network_trace[self.network_seg_index]*self.remaining_buffer
						self.remaining_time -= self.remaining_buffer
						self.downloadedPartialVideo = 1
						self.request_pre = 0
						self.buffer_size_EL = max(0.0, np.round(self.buffer_size_EL - self.remaining_buffer))
						if self.buffer_size_EL > 0:
							self.remaining_buffer = 1
						else:
							self.remaining_buffer = 0
						if self.remaining_time == 0:
							self.remaining_time = 1
						if self.buffer_size_EL == 0 :
							self.freezing = 1
						if IS_CORRECT and self.retransmit_available(network_trace_aux, self.video_download_timestamp - CORRECT_TIME_HEAD):
							print("We are going to correct:", int(np.round(self.display_time)), self.display_time)
							repair = self.correct(yaw_trace, pitch_trace, rate_cut, network_trace_aux)
	
							# print("Going into freezing")
						print("NOT FREEZING, time > buffer, incomplete")
						print("buffer size:", self.buffer_size_EL)
				# print("NOT freezing(END), buffer size: %f, current time: %f" % (self.buffer_size_EL, self.video_download_timestamp))

			## If it's freezing time
			else:
				assert (self.remaining_buffer == 0)
				assert (self.buffer_size_EL == 0)
				print("In freezing, time is: %f, buffer: %f, remaining buffer:%f,  el index: %f" % \
						(self.video_download_timestamp, self.buffer_size_EL, self.remaining_buffer, self.video_seg_index_EL))
				if network_trace[self.network_seg_index]*self.remaining_time >= self.video_segment:
					if self.video_version >= 0:
						self.video_download_timestamp += self.video_segment/(network_trace[self.network_seg_index])
						# if self.freezing:
						# 	self.freezing_time_current_chunk += self.video_segment/(network_trace[self.network_seg_index])
						self.remaining_time -= self.video_segment/(network_trace[self.network_seg_index])
						# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
						self.remaining_buffer += 1
						self.buffer_size_EL += 1
						self.downloadedPartialVideo = 0
						self.freezing_time_current_chunk += self.video_segment/(network_trace[self.network_seg_index])
						self.total_freezing += self.freezing_time_current_chunk

						# temprol_eff = 1 - min(1, max(self.video_download_timestamp - self.video_seg_index_EL, 0))
						# if self.video_seg_index_EL > int(np.floor(self.video_download_timestamp)):
						# 	self.buffer_size_EL += 1
						# elif self.video_seg_index_EL == int(np.floor(self.video_download_timestamp)):
						# 	self.buffer_size_EL += (np.ceil(self.video_download_timestamp) - self.video_download_timestamp)
						self.EVR_EL_Recordset.append([self.video_seg_index_EL, self.video_version, network_trace[self.network_seg_index], \
							self.yaw_predict_value_quan*30-15, yaw_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
							self.pitch_predict_value_quan, pitch_trace[self.video_seg_index_EL*VIDEO_FPS+VIDEO_FPS/2],\
							VP_SPAN_YAW, VP_SPAN_PITCH, RETRAN_UTILI_INIT, self.freezing_time_current_chunk])
						# self.video_seg_index_EL = max(self.video_seg_index_EL + 1, int(np.ceil(self.video_download_timestamp)))
						self.video_seg_index_EL += 1		
						self.freezing_time_current_chunk = 0
						self.freezing = 0	
						print("change from freezing to UNFREEZING")		
						print("new index:%f" % self.video_seg_index_EL)	
						#### Liyang
						# video_rate = rate_cut[self.video_version] - rate_cut[0]
						####
					else :
						print("Unknown video version.")
						# self.remaining_time = 1
						# self.downloadedPartialVideo = 0
						# self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
						# self.video_segment = 0
						# self.freezing_time_current_chunk += self.remaining_time

				## in freezing, and remaining time is not enough
				else:
					self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8))
					self.video_segment = self.video_segment - network_trace[self.network_seg_index]*self.remaining_time
					self.remaining_time = 1
					self.downloadedPartialVideo = 1
					self.freezing_time_current_chunk += self.remaining_time


			# if not self.downloadedPartialVideo:
			# 	if self.video_segment >= 0 :
			# 		if len(self.video_bw_history) == 0:
			# 			bw = video_rate/self.video_download_timestamp 
			# 		else :
			# 			bw = video_rate/(self.video_download_timestamp - self.video_bw_history[-1][1])
					
			# 		self.video_bw_history.append([bw, self.video_download_timestamp, self.video_version])
			# 		# print(bw,network_trace[self.network_seg_index], int(np.floor(temp_video_download_timestamp)))
			# 	else:
			# 		self.video_bw_history.append([network_trace[self.network_seg_index], self.video_download_timestamp, -1])

		# else:
		# 	if self.video_version == 0:
		# 		print('Download base tier, bandwidth is not enough.')
		# 	elif self.video_version >= 1:
		# 		print('Download enhancement tier, bandwidth is not enough.')
		# 	self.video_download_timestamp = np.ceil(self.video_download_timestamp + (10.**-8)) + self.request_pre
		# 	self.video_segment = self.video_segment - network_trace[self.network_seg_index]*self.remaining_time
		# 	self.remaining_time = 1 - self.request_pre
		# 	self.downloadedPartialVideo = 1
		# 	self.request_pre = 0
		# 	self.network_seg_index += 1

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

		# self.buffer_size_BL = max(0, self.buffer_size_BL - (self.video_download_timestamp - temp_video_download_timestamp))
		# preivous_buffer = self.buffer_size_EL
		# self.buffer_size_EL = max(0, self.buffer_size_EL - (self.video_download_timestamp - temp_video_download_timestamp))

		if np.floor(self.video_download_timestamp) != np.floor(temp_video_download_timestamp):
			# if self.buffer_size_BL == 0:
			# 	self.GBUFFER_BL_EMPTY_COUNT += 1
			# 	self.video_seg_index_BL = int(np.floor(self.video_download_timestamp)) ## start to fast retransmit BL video
			# if self.buffer_size_EL == 0:
			# 	self.GBUFFER_EL_EMPTY_COUNT += 1
				# self.video_seg_index_EL = int(np.floor(self.video_download_timestamp))
			bw = network_trace[int(np.floor(temp_video_download_timestamp))]
			self.video_bw_history.append([bw, self.video_download_timestamp])
			self.buffer_size_history.append(self.buffer_size_EL)
			self.network_seg_index += 1

		return temp_display_time, repair


	## Check 5G availability
	def retransmit_available(self, network_trace_aux, video_download_timestamp):
		availability = network_trace_aux[int(np.floor(video_download_timestamp))]
		if availability != 0:
			return True
		else:
			return False

	def repair(self, correct_time, correct_index, yaw_trace, pitch_trace, rate_cut, network_trace_aux):
		repair = 0
		if self.correctness_using == 0 and correct_index < 600:	
			# print("Repair:", correct_index)	
			correct_time -= REPAIR_TIME_HEAD
			new_yaw_predict_value, new_yaw_predict_value_quan = self.predict_yaw(yaw_trace, correct_time, correct_index)
			new_pitch_predict_value, new_pitch_predict_value_quan = self.predict_pitch(pitch_trace, correct_time, correct_index)
			## Retransmit the how chunk as previous 

			extra_segment = rate_cut[MAX_VIDEO_VERSION]
			downloading_5G = 0
			# new_temprol_eff = 0
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
					self.addition_time += (REPAIR_TIME_HEAD)
					return	
				downloading_5G = (extra_segment - retran_bandwidth_1*REPAIR_TIME_HEAD)/retran_bandwidth_2
				if downloading_5G > REPAIR_TIME_THRES:
					print("5G Too bad")
					self.correctness_using = 0
					self.addition_time += (REPAIR_TIME_HEAD + REPAIR_TIME_THRES)
					return
				# new_temprol_eff = min(1, 1 - downloading_5G)
			# else:
			# 	new_temprol_eff = 1
			if len(self.EVR_EL_Recordset) > 0:
				assert self.EVR_EL_Recordset[-1][0] < correct_index
			print("Downling 5g time:", downloading_5G)
			self.addition_time += (REPAIR_TIME_HEAD + downloading_5G)
			# self.buffer_size_EL += 1

			# self.EVR_EL_Recordset.append([correct_index, MAX_VIDEO_VERSION, (retran_bandwidth_1+retran_bandwidth_2)/2, \
			# 	new_yaw_predict_value_quan*30-15, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
			# 	new_pitch_predict_value_quan, pitch_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
			# 	VP_SPAN_YAW, VP_SPAN_PITCH, extra_segment, downloading_5G])
			# print("repair degree",new_yaw_predict_value_quan*30-15, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2])
			# print("Effective time", new_temprol_eff)

			

			## If arrive here, means 5G has already handle the freezing, go back into normal

			self.video_download_timestamp += downloading_5G
			# if self.freezing:
			# 	self.freezing_time_current_chunk += self.video_segment/(network_trace[self.network_seg_index])
			self.remaining_time -= downloading_5G
			if self.remaining_time == 0:
				self.remaining_time = 1
			# self.remaining_time = 1 - (self.video_download_timestamp) + np.floor(self.video_download_timestamp)
			self.remaining_buffer += 1
			self.buffer_size_EL += 1
			self.downloadedPartialVideo = 0
			self.freezing_time_current_chunk += downloading_5G
			self.total_freezing += self.freezing_time_current_chunk

			# temprol_eff = 1 - min(1, max(self.video_download_timestamp - self.video_seg_index_EL, 0))
			# if self.video_seg_index_EL > int(np.floor(self.video_download_timestamp)):
			# 	self.buffer_size_EL += 1
			# elif self.video_seg_index_EL == int(np.floor(self.video_download_timestamp)):
			# 	self.buffer_size_EL += (np.ceil(self.video_download_timestamp) - self.video_download_timestamp)
			self.EVR_EL_Recordset.append([correct_index, MAX_VIDEO_VERSION, (retran_bandwidth_1+retran_bandwidth_2)/2, \
				new_yaw_predict_value_quan*30-15, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
				new_pitch_predict_value_quan, pitch_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
				VP_SPAN_YAW, VP_SPAN_PITCH, extra_segment, downloading_5G])
			# self.video_seg_index_EL = max(self.video_seg_index_EL + 1, int(np.ceil(self.video_download_timestamp)))
			self.video_seg_index_EL += 1		
			self.freezing_time_current_chunk = 0
			self.freezing = 0	

			self.record_info = np.append(self.record_info, \
					[self.video_download_timestamp, self.buffer_size_EL,
					new_yaw_predict_value, yaw_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2],\
					new_pitch_predict_value, pitch_trace[correct_index*VIDEO_FPS+VIDEO_FPS/2], correct_time - REPAIR_TIME_THRES])
			self.correctness_using = 0
			print("change from freezing to UNFREEZING, through 5G")		
			print("new index:%f" % self.video_seg_index_EL)	
			repair = 1

		return repair


	## To use the second 5G high-speed low-latency connection to correct vp
	def correct(self,yaw_trace, pitch_trace, rate_cut, network_trace_aux):
		repair = 0
		if self.correctness_using == 0:
			correct_time = np.floor(self.display_time)
			correct_index = int(np.round(self.display_time))
			next_EL = []
			while self.correct_ptr < len(self.EVR_EL_Recordset) and self.EVR_EL_Recordset[self.correct_ptr][0] < correct_index :
				print("Whether repair %f %f" %(self.EVR_EL_Recordset[self.correct_ptr][0], correct_index))
				self.correct_ptr += 1
			print("Final get %f %f" % (self.EVR_EL_Recordset[-1][0], correct_index))
			print(len(self.EVR_EL_Recordset), self.correct_ptr, correct_index)
			if self.correct_ptr >= len(self.EVR_EL_Recordset):
				if IS_REPAIR and correct_index < VIDEO_LEN:
					assert(self.freezing == 1)
					# print("Repair:", correct_index)
					# if len(self.EVR_EL_Recordset) > 0:
					# 	print("last one",self.EVR_EL_Recordset[-1][0])
					repair = self.repair(correct_time, correct_index, yaw_trace, pitch_trace, rate_cut, network_trace_aux)
				return repair
			else:
				correct_time -= CORRECT_TIME_HEAD
				next_EL = self.EVR_EL_Recordset[self.correct_ptr]
				assert (next_EL[9] == 0)
				assert (next_EL[10] == 0)
				# print(next_EL[0])
				if next_EL[0] > correct_index:
					# No need to correct
					print("Too far away, not yet.")
					return repair
				elif next_EL[0] == correct_index:

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
						return repair
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
						# assert (next_EL[1] >= 1)

						## Calculate size of data should be retransmitted
						retran_data_size = (RETRAN_EXTRA*total_tiles/MAX_TILES)*(rate_cut[1]-rate_cut[0])
						# print(retran_data_size)

						## Simulate 5G retransmisstion and correct
						downloading_5G = 0
						## Start to retransmit
						retran_bandwidth_1 = network_trace_aux[correct_index-1]
						retran_bandwidth_2 = network_trace_aux[correct_index]						
						self.correctness_using = 1
						

						if retran_data_size > retran_bandwidth_1*CORRECT_TIME_HEAD:
							if retran_bandwidth_2 == 0:
								self.EVR_EL_Recordset[self.correct_ptr][10] = retran_data_size*CORRECT_TIME_HEAD
								self.addition_time += CORRECT_TIME_HEAD
								self.correctness_using = 0
								return repair
							downloading_5G = (retran_data_size - retran_bandwidth_1*CORRECT_TIME_HEAD)/retran_bandwidth_2
							if downloading_5G >= CORRECT_TIME_THRES:
								self.EVR_EL_Recordset[self.correct_ptr][10] = retran_data_size*CORRECT_TIME_THRES
								self.correctness_using = 0
								self.addition_time += (CORRECT_TIME_HEAD+CORRECT_TIME_THRES)
								print("Second connection too bad!!!")
								return repair						

						self.addition_time += (CORRECT_TIME_HEAD+downloading_5G)
						# new_temprol_eff = 1 - downloading_5G
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
							self.EVR_EL_Recordset[self.correct_ptr][9] = retran_data_size
							if self.freezing:
								self.EVR_EL_Recordset[self.correct_ptr][10] = downloading_5G
								print("Must be problem!! Cannot be triggered!")
							self.correctness_using = 0
							print("Update EL", new_central, new_yaw_span, retran_data_size, self.EVR_EL_Recordset[self.correct_ptr][10])
						else:
							## Liyang, should modify to update EL recordest
							return repair
							## 5G is not usable
						# print("Could correct!")
						# print("previous central", central)
						# print("new central", new_yaw_predict_value)
						# print("direction", direction_central)
						# print("------------------------------")
		return repair 


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
	network_trace_aux = load_5G.load_5G_Data(NETWORK_TRACE_FILENAME, EXTRA_MULTIPLE, EXTRA_ADD)
	yaw_trace, pitch_trace = loadViewportTrace()
	network_pdf, pdf_bins, pdf_patches = plot_pdf(network_trace)
	network_cdf, cdf_bins, cdf_patches = plot_cdf(network_trace)
	rate_cut = rate_determine(network_cdf)
	video_trace = loadVideoTrace(rate_cut)
	streaming_sim = streaming()
	streaming_sim.run(network_trace, yaw_trace, pitch_trace, video_trace, rate_cut,network_trace_aux)
	# print(len(streaming_sim.buffer_size_history))
	if IS_DEBUGGING:
		display(streaming_sim.record_info, streaming_sim.EVR_EL_Recordset, \
			rate_cut, yaw_trace, pitch_trace, network_trace, streaming_sim.bw_info)
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
	mat_contents = sio.loadmat(VIEWPORT_TRACE_FILENAME1)
	yaw_trace_data = mat_contents['view_angle_yaw_combo'] + 180 # array of structures
	pitch_trace_data = mat_contents['view_angle_pitch_combo'] + 90 # array of structures
	# print(len(yaw_trace_data[0]), len(pitch_trace_data[0]))
	assert (len(yaw_trace_data[0]) > VIDEO_LEN*VIDEO_FPS and len(pitch_trace_data[0]) > VIDEO_LEN*VIDEO_FPS)
	return yaw_trace_data[0][:VIDEO_LEN*VIDEO_FPS], pitch_trace_data[0][:VIDEO_LEN*VIDEO_FPS]

def loadVideoTrace(rate_cut):
	# video_trace0 = rate_cut[0] * np.ones(VIDEO_LEN)
	video_trace1 = rate_cut[0] * np.ones(VIDEO_LEN)
	video_trace2 = rate_cut[1] * np.ones(VIDEO_LEN)
	video_trace3 = rate_cut[2] * np.ones(VIDEO_LEN)
	return [video_trace1, video_trace2, video_trace3]

def plot_pdf(trace):
	global FIGURE_NUM	
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	n, bins, patches = plt.hist(trace, range(0, int(np.ceil(max(trace))) + 1), normed = 1, label='PDF', color='b')
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def plot_cdf(trace):
	global FIGURE_NUM
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	n, bins, patches = plt.hist(trace, range(0,int(np.ceil(max(trace))) + 1), normed = 1, cumulative=True, label='CDF', histtype='stepfilled', color='b')
	# print(a,b,c)
	if IS_DEBUGGING: f.show()
	return n, bins, patches

def rate_determine(network_cdf):
	cut_percent = [0.4, 0.6, 0.8]
	rate_cut = []
	cut_idx = 0
	for cum in network_cdf:
		if cum >= np.max(network_cdf) * cut_percent[cut_idx]:
			rate_cut.append(np.round(network_cdf.tolist().index(cum)))
			cut_idx += 1
			if cut_idx >= len(cut_percent): 
				break
	if IS_DEBUGGING: 
		print('Rate1 = %f, Rate2 = %f, Rate3 = %f' % (rate_cut[0],rate_cut[1],rate_cut[2]))
	return rate_cut

def display(record_info, EVR_EL_Recordset, rate_cut, yaw_trace, pitch_trace, network_trace, bw_info):
	# print(len(record_info))
	# print(len(EVR_EL_Recordset))
	# print(len(EVR_BL_Recordset))
	# record_info = np.array(record_info)
	display_result = record_info.reshape(len(record_info)/7, 7).T
	bw_result = bw_info.reshape(len(bw_info)/2, 2).T
	global FIGURE_NUM
	f = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	# plt.plot(display_result[0], display_result[1],'r-', label='Base Tier Buffer Length')
	plt.plot(display_result[0], display_result[1],'b-', label='Enhancement Tier Buffer Length')
	plt.legend(loc='upper right')
	plt.title('ET Buffer Length')
	plt.xlabel('Second')
	plt.ylabel('Second')
	plt.axis([0, 600, 0, 20])

	# plt.ylim(-1.5, 2.0)
	h = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	plt.plot(display_result[6], display_result[2],'b-', label='Predict Viewport (horizontal)')
	plt.plot(display_result[6], display_result[3],'r-', label='Real Viewport (horizontal)')
	plt.legend(loc='upper right')
	plt.title('Viewport Predict and Real Trace')
	plt.xlabel('Second')
	plt.ylabel('Degree')
	plt.axis([0, 600, 0, 450])

	p = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	plt.plot(range(1,int(bw_result[1][-1])+1), network_trace[:int(bw_result[1][-1])],'r-',label='Real Bandwidth')
	plt.legend(loc='upper right')
	plt.title('Bandwidth Predict and Real Trace')
	plt.xlabel('Second')
	plt.ylabel('Mbps')
	plt.axis([0, 600, 0, max(network_trace)+30])

	display_bitrate = [0.0]*VIDEO_LEN
	receive_bitrate = [0.0]*VIDEO_LEN
	extra_cost = [0.0]*VIDEO_LEN
	extra_freezing = [0.0]*VIDEO_LEN
	# print(EVR_BL_Recordset)
	# for i in range (0,BUFFER_BL_INIT):
	# 	display_bitrate[i] += rate_cut[0]/6
	# 	receive_bitrate[i] += rate_cut[0]/6
	display_bitrate[0] += rate_cut[2]
	receive_bitrate[0] += rate_cut[2]
	# for i in range(0, len(EVR_BL_Recordset)):
	# 	display_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]
	# 	receive_bitrate[EVR_BL_Recordset[i][0]] += rate_cut[EVR_BL_Recordset[i][1]]

	# print(EVR_EL_Recordset)
	total_eff = 0.0
	for i in range(0,len(EVR_EL_Recordset)):
		yaw_distance = 0.
		eff= 0.
		sum_eff = 0.
		for j in range(0, VIDEO_FPS):
			yaw_distance = min(np.abs(yaw_trace[EVR_EL_Recordset[i][0]*VIDEO_FPS+j] - (EVR_EL_Recordset[i][3])), \
							360 - np.abs(yaw_trace[EVR_EL_Recordset[i][0]*VIDEO_FPS+j] - (EVR_EL_Recordset[i][3])))
			eff = min(1, max(0, (((EVR_EL_Recordset[i][7] + USER_VP)/2) - yaw_distance)/USER_VP))
			sum_eff += eff
		sum_eff /= VIDEO_FPS
		total_eff += sum_eff
		display_bitrate[EVR_EL_Recordset[i][0]] += sum_eff*rate_cut[EVR_EL_Recordset[i][1]]
		receive_bitrate[EVR_EL_Recordset[i][0]] += rate_cut[EVR_EL_Recordset[i][1]]
		extra_freezing[EVR_EL_Recordset[i][0]] = EVR_EL_Recordset[i][10]
		extra_cost[EVR_EL_Recordset[i][0]] = EVR_EL_Recordset[i][9]
	average_eff = total_eff/len(EVR_EL_Recordset)
	print("Total extra cost:", sum(extra_cost))

		# print (sum_eff, EVR_EL_Recordset[i][0])
		# print(EVR_EL_Recordset[i][3]*30-15, EVR_EL_Recordset[i][3], EVR_EL_Recordset[i][5],EVR_EL_Recordset[i][9])
	# return
	print("Effective bitrate:", sum(display_bitrate))
	print("Average coverage ratio:", average_eff)
	g = plt.figure(FIGURE_NUM)
	FIGURE_NUM += 1
	plt.plot(range(1,VIDEO_LEN+1), display_bitrate, 'b-', label='Effective Video Bitrate')
	plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-', label='Received Video Bitrate')
	plt.legend(loc='upper right')
	plt.title('Effective & Received Video Bitrate')
	plt.xlabel('Second')
	plt.ylabel('Mbps')
	plt.axis([0, 600, 0, max(receive_bitrate)+30])

	if IS_CORRECT:
		q = plt.figure(FIGURE_NUM)
		FIGURE_NUM += 1		
		plt.bar(range(1,VIDEO_LEN+1), extra_freezing, label='Freezing Time',edgecolor='blue')
		# plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-', label='Received Video Bitrate')
		plt.legend(loc='best')
		plt.title('Freezing Time')
		plt.xlabel('Second')
		plt.ylabel('Second')
		plt.axis([0, 600, 0, max(extra_freezing)+0.1])

		r = plt.figure(FIGURE_NUM)
		FIGURE_NUM += 1		
		plt.bar(range(1,VIDEO_LEN+1), extra_cost, label='Extra Cost', edgecolor='blue')
		# plt.plot(range(1,VIDEO_LEN+1), receive_bitrate, 'r-', label='Received Video Bitrate')
		plt.legend(loc='best')
		plt.title('Extra Cost')
		plt.xlabel('Second')
		plt.ylabel('Mb')
		plt.axis([0, 600, 0, max(extra_cost)+ 5])

	# i = plt.figure(5)
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
		r.show()

	if IS_SAVING:
		f.savefig('VP_ONLY_BT_&_ET_Buffer_Length.eps', format='eps', dpi=1000)
		g.savefig('VP_ONLY_Effective_Received_Video_Bitrate.eps', format='eps', dpi=1000)
		h.savefig('VP_ONLY_Viewport_Predict_&_Real_Trace.eps', format='eps', dpi=1000)
		p.savefig('VP_ONLY_Bandwidth_Predict_&_Real_Trance.eps', format='eps', dpi=1000)
		q.savefig('VP_ONLY_Extra_Cost.eps', format='eps', dpi=1000)
	return

if __name__ == '__main__':
	main()
	
