# Do dynamic rate allocation at fix frequency
# And also for two tier without dynamic allocation

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G
import math
import utilities as uti
import comparison
import cmp_test
import pickle


DO_DYNAMIC = 1				# <=========== Switch between stati two tier and fix periodic optimization
CODING_TYPE = 2				# <=========== For NLC in JETCAS
VIDEO_LEN = 300
VIDEO_FPS = 30
UPDATE_FREQUENCY = 30
# BW trace
REGULAR_CHANNEL_TRACE = './traces/bandwidth/BW_Trace_5G_5.txt'  
# Don't change following -----------------------------------------------#
if REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_5.txt':		#
	VIDEO_LEN = 450														#
# Don't change following -----------------------------------------------#

# DELAY_TRACE = 'delay_1.txt'
REGULAR_MULTIPLE = 1
REGULAR_ADD = 0
# VP trace

# For JETCAS revision
REVISION = 1
if not REVISION:
	VIEWPORT_TRACE_FILENAME_NEW = './traces/output/Video_13_alpha_beta_new.mat'    ##  9 for 1,  13 for 2
	USER = -1
else:
	VIEWPORT_TRACE_FILENAME_NEW = './traces/output/gt_theta_phi_vid_3.p'    ##  0 for 1,  6 for 2
	USER = 0				#  <=====================    Change here to control the fov trace for REVISION
CMP_VP_TRACE_FILENAME = './traces/output/gt_theta_phi.p'

# System parameters
BUFFER_BL_INIT = 10
BUFFER_EL_INIT = 1
Q_REF_BL = 10
Q_REF_EL = 1
ET_MAX_PRED = Q_REF_EL + 1

CHUNK_DURATION = 1.0
#Others
KP = 0.6		# P controller
KI = 0.01		# I controller
PI_RANGE = 10
DELAY = 0.01		# in second
PI_SMOOTH = 1

# For alpha/gamma
BUFFER_RANGE = 4
BW_DALAY_RATIO = 0.95

# FoR MMSys'19 comparison, single tp trace detail info show
COMPARISON = 1					# For JETCAS, disable this value
if COMPARISON:
	HITRATE_DICT = './traces/hitrate/hitrate.p'
	UPDATE_FREQUENCY = 5
	REGULAR_CHANNEL_TRACE = './traces/bandwidth/NEW_MIX_17.txt'  
	VIDEO_LEN = 45
	DELAY = 0.01	# in second


class Streaming(object):
	def __init__(self, network_trace, yaw_trace, pitch_trace, video_trace, rate_cut, optimal_buffer_length, alpha_idx, gamma_idx, hitrate):
		self.dy_type = 'fix'
		self.fov_file = alpha_idx
		self.network_file = gamma_idx
		self.dynamic = DO_DYNAMIC
		self.network_trace = network_trace
		self.yaw_trace = yaw_trace
		self.pitch_trace = pitch_trace
		self.video_trace = video_trace

		self.rate_cut = []
		self.rate_cut.append(rate_cut)
		self.rate_cut_version = 0
		self.rate_cut_time = [0.0]

		self.network_ptr = 0
		self.network_time = 0.0
		self.display_time = 0.0

		self.video_seg_index_bl = BUFFER_BL_INIT 
		self.video_seg_index_el = BUFFER_EL_INIT
		self.buffer_size_bl = BUFFER_BL_INIT
		self.buffer_size_el = BUFFER_EL_INIT
		self.buffer_history = []
		if not DO_DYNAMIC:
			# For generating gamma curve
			# self.target_et_buffer = Q_REF_EL

			# For static test
			self.target_et_buffer = optimal_buffer_length
		else:
			self.target_et_buffer = optimal_buffer_length

		self.upper_et_buffer = self.target_et_buffer + 2
		self.download_partial = 0
		self.video_seg_size = 0.0

		# self.bw_info = []
		self.video_bw_history = []
		self.video_version = 0
		self.video_seg_index = 0
		self.target_buffer_history = []
		self.target_buffer_history.append(Q_REF_EL)

		self.yaw_predict_value = 0.0
		self.yaw_predict_quan = 0
		self.pitch_predict_value = 0.0
		self.pitch_predict_quan = 0

		# self.record_info = []
		self.evr_bl_recordset = []
		self.evr_el_recordset = []
		self.bl_freezing_count = 0
		self.el_freezing_count = 0
		self.freezing_time = 0

		# alpha and gamma
		self.alpha_history = []
		self.hitrate = hitrate


	def run(self):
		while self.video_seg_index_bl < VIDEO_LEN or \
			(self.video_seg_index_bl >= VIDEO_LEN and self.video_seg_index_el < VIDEO_LEN):
			if not self.download_partial:
				# adaptive rate allocation at fix frequency
				if DO_DYNAMIC and int(np.floor(self.display_time/UPDATE_FREQUENCY)) != self.rate_cut_version:
					# print("optimization, current network ptr is: %s, display_time: %s" % (self.network_ptr, self.display_time))
					self.alpha_history[-1][2] = self.rate_cut_version + 1
					average_bw, std_bw, average_bw_real = uti.get_average_bw(self.display_time, self.video_bw_history, self.rate_cut_version)
					if not COMPARISON:
						new_rate_cut, new_target_et_buffer = uti.rate_optimize(self.display_time, \
														average_bw, std_bw, self.alpha_history,\
														self.rate_cut_version, VIEWPORT_TRACE_FILENAME_NEW)
					else:
						new_rate_cut, new_target_et_buffer = comparison.cmp_rate_optimize(self.display_time, \
														average_bw, std_bw, self.alpha_history,\
														self.rate_cut_version, CMP_VP_TRACE_FILENAME)
					self.rate_cut.append(new_rate_cut)
					self.rate_cut_version += 1
					self.rate_cut_time.append(self.display_time)

					self.target_et_buffer = new_target_et_buffer
					self.upper_et_buffer = self.target_et_buffer + 2
					self.target_buffer_history.append(new_target_et_buffer)
				# Finish PI control
				sniff_bw = uti.predict_bw(self.video_bw_history)
				# self.bw_info = np.append(self.bw_info, [sniff_bw, self.network_time])
				self.PI_control(sniff_bw)
				self.update_seg_size()
			
				# temp_index = self.video_seg_index
				if self.video_version != 0:
					assert self.video_version != -1
					self.yaw_predict_value, self.yaw_predict_quan = uti.predict_yaw_trun(self.yaw_trace, self.display_time, self.video_seg_index)
					self.pitch_predict_value, self.pitch_predict_quan = uti.predict_pitch_trun(self.pitch_trace, self.display_time, self.video_seg_index)
					if COMPARISON:
						hit_time = int(np.round(self.display_time))
						# print(hit_time, self.display_time)
						pred_interval = self.video_seg_index - hit_time
						assert pred_interval >= 0
						self.chunk_hit = self.hitrate[hit_time][pred_interval*VIDEO_FPS:(pred_interval+1)*VIDEO_FPS]
			previous_time, recording_el = self.fetching()
			if round(self.display_time,4) >= VIDEO_LEN:
				break
			# Record EL 
			if not self.download_partial and recording_el:
				print("need recoding el")


	def fetching(self):
		temp_video_display_time = self.display_time
		new_temp_video_display_time = temp_video_display_time
		video_chunk_size = 0.0
		recording_el = 0
		is_same_el = 0
		record_bw = 1
		rebuf = 0.0

		# Before sending request, check whether EL is requested and EL buffer is greater than MAX
		if self.video_version >= 1 and round(self.buffer_size_el, 3) > self.upper_et_buffer:
			assert self.download_partial != 1
				# print("Current tiem is %s, and buffer length is %s, %s, is downloading %s" %(self.display_time, self.buffer_size_bl, self.buffer_size_el, self.video_version))
			# need to sleep, nothing should be recorded
			recording_el = 0
			#Calculate sleep time
			sleep_time = self.video_seg_index_el - self.upper_et_buffer - temp_video_display_time
			if sleep_time > 1:
				print("Buffer change trigger sleep greater than 1s, current time is: %s and sleep time is: %s" %(self.display_time, sleep_time))
			first_sleep = sleep_time - np.floor(sleep_time - 10.**-8)
			sleep_time -= first_sleep
			# print("sleep_time is: %s" % sleep_time)
			self.display_time += first_sleep 
			self.network_time += first_sleep
			self.network_ptr = int(np.floor(self.network_time))

			if round(self.display_time, 2) != round(self.network_time, 2):
				print("Network time and display tims is not sync")
			if round(self.video_seg_index_el - self.upper_et_buffer, 2) != round(self.display_time, 2):
				print("Two methods for sleep are not sync")

			if round(self.video_seg_index_el - temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
				self.buffer_size_el = np.minimum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
			else:
				if round(self.video_seg_index_el - temp_video_display_time, 2) != round(self.buffer_size_el, 2)  :
					print("not equal case 1",self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
				self.buffer_size_el = np.maximum(self.buffer_size_el - first_sleep, 0.0)
			if not round(self.network_time, 3).is_integer():
				print("Current network time is %s, not integer" % self.network_time)
			
			self.buffer_size_bl -= first_sleep
			self.video_bw_history.append([self.network_trace[self.network_ptr], self.network_time, -1, \
									self.rate_cut_version, self.network_trace[self.network_ptr]])
			# print("after sleep, Current tiem is %s, and buffer length is %s, %s, is downloading %s" %(self.display_time, self.buffer_size_bl, self.buffer_size_el, self.video_version))
			self.buffer_history.append([round(round(self.buffer_size_bl*100 + 2)/100), round(round(self.buffer_size_el*100 + 2)/100), self.display_time])
			if self.display_time >= 2.0:
				self.calculate_alpha()
				self.alpha_history.append([self.display_time-1, uti.record_alpha(self.yaw_trace, self.pitch_trace, self.display_time, VIDEO_LEN),\
										self.rate_cut_version])
			temp_video_display_time = self.display_time
			if sleep_time > 0:
				print("greater than 1s, sleep left is: %s" % sleep_time)
				assert sleep_time.is_integer()
				for i in range(int(sleep_time)):
					self.display_time += CHUNK_DURATION 
					self.network_time += CHUNK_DURATION
					self.network_ptr = int(np.floor(self.network_time))	

					if round(self.video_seg_index_el - temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
						self.buffer_size_el = np.minimum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
					else:
						if round(self.video_seg_index_el - temp_video_display_time, 2) != round(self.buffer_size_el, 2)  :
							print("not equal case 1",self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
						self.buffer_size_el = np.maximum(self.buffer_size_el - CHUNK_DURATION, 0.0)

					if not round(self.network_time, 3).is_integer():
						print("Current network time is %s, not integer" % self.network_time)
					print("el buffer is: %s, time is %s, " % (self.buffer_size_el, self.display_time))

					self.buffer_size_bl -= CHUNK_DURATION
					self.video_bw_history.append([self.network_trace[self.network_ptr], self.network_time, -1, \
											self.rate_cut_version, self.network_trace[self.network_ptr]])
					# print("after sleep, Current tiem is %s, and buffer length is %s, %s, is downloading %s" %(self.display_time, self.buffer_size_bl, self.buffer_size_el, self.video_version))
					self.buffer_history.append([round(round(self.buffer_size_bl*100 + 2)/100), round(round(self.buffer_size_el*100 + 2)/100), self.display_time])
					if self.display_time >= 2.0:
						self.calculate_alpha()		
						self.alpha_history.append([self.display_time-1, uti.record_alpha(self.yaw_trace, self.pitch_trace, self.display_time, VIDEO_LEN),\
												self.rate_cut_version])
					temp_video_display_time = self.display_time
			return temp_video_display_time, recording_el

		# All below don't need to consider sleep!
		if not self.download_partial:
			current_time_left = np.ceil(temp_video_display_time + 10.**-8) - temp_video_display_time
			
			# For the delay part, need record
			if current_time_left < DELAY:
				print("Delay larger than current time left, tiem left is %s" % current_time_left)
				self.buffer_history.append([round(round(self.buffer_size_bl*100 + 2)/100), round(round(self.buffer_size_el*100 + 2)/100), self.display_time + current_time_left])


				if self.display_time + DELAY >= 2.0:
					self.calculate_alpha()
					self.alpha_history.append([int(self.display_time+DELAY)-1, uti.record_alpha(self.yaw_trace, self.pitch_trace, self.display_time, VIDEO_LEN),\
										self.rate_cut_version])
			self.network_time += DELAY
			self.network_ptr = int(self.network_time)

			# No freezing
			if self.buffer_size_bl >= DELAY:
				self.display_time += DELAY
				self.buffer_size_bl -= DELAY
				if round(self.video_seg_index_el - temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
					self.buffer_size_el = np.minimum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
				else:
					if round(self.video_seg_index_el - temp_video_display_time, 2) != round(self.buffer_size_el, 2) and self.buffer_size_el != 0:
						print("not equal case 2", self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
					self.buffer_size_el = np.maximum(self.buffer_size_el - DELAY, 0.0)
			else:
				assert self.buffer_size_el == 0
				print("will not happen in this simulation!!!!!!!!!!!!!!!!!!!!!")


			## Time after request
			new_temp_video_display_time = self.display_time

		# print("current network ptr is: %s, display_time: %s, index %s, %s" % (self.network_ptr, self.display_time, self.video_seg_index_bl, self.video_seg_index_el))
		throughput = self.network_trace[self.network_ptr]
		duration = np.ceil(self.network_time + 10.**-8) - self.network_time
		if throughput * duration >= self.video_seg_size:
			download_duration =  self.video_seg_size/throughput
			self.download_partial = 0

			self.network_time += download_duration
			self.network_ptr = int(self.network_time)
			self.display_time += np.minimum(self.buffer_size_bl, download_duration)
			
			# how display time change and how buffer change (decrease phase)
			rebuf = np.maximum(download_duration - self.buffer_size_bl, 0.0)

			self.buffer_size_bl = np.maximum(self.buffer_size_bl - download_duration, 0.0)
			if round(self.video_seg_index_el - new_temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el > 0:
				self.buffer_size_el = np.minimum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
			else:
				if round(self.video_seg_index_el - new_temp_video_display_time, 2) != round(self.buffer_size_el, 2) and self.buffer_size_el != 0:
					print("not equal case 3", self.video_seg_index_el, new_temp_video_display_time, self.buffer_size_el)
				self.buffer_size_el = np.maximum(self.buffer_size_el - download_duration, 0.0)
			
			## Switch based on version, go through the buffer and time changing again
			## And finish the 

			# Increse buffer phase
			if self.video_version == 0:
				self.evr_bl_recordset.append([self.video_seg_index_bl, self.video_version, self.display_time,\
											 self.network_time, self.network_ptr, self.rate_cut_version])
				self.video_seg_index_bl += int(CHUNK_DURATION)
				self.video_seg_index_el = np.maximum(int(np.floor(self.display_time))+1, self.video_seg_index_el)
				self.buffer_size_bl += CHUNK_DURATION

			elif self.video_version >= 1:
				temporal_eff = 1.0
				if self.buffer_size_el > self.upper_et_buffer:
					print(self.buffer_size_el, self.upper_et_buffer)
				if self.video_seg_index < int(np.floor(self.display_time)):
					## This chunk is not useful
					assert self.buffer_size_el == 0
					self.video_seg_index_el = int(np.floor(self.display_time)) + 1
					temporal_eff = 0.0
				elif self.video_seg_index == int(np.floor(self.display_time)):
					assert round(self.buffer_size_el,3) == 0
					self.video_seg_index_el += int(CHUNK_DURATION)
					self.buffer_size_el += CHUNK_DURATION - (self.display_time - int(np.floor(self.display_time)))
					temporal_eff = CHUNK_DURATION - (self.display_time - np.floor(self.display_time))
				else:
					# 
					self.video_seg_index_el += int(CHUNK_DURATION)
					self.buffer_size_el += CHUNK_DURATION
					temporal_eff = 1.0
				
				self.evr_el_recordset.append([self.video_seg_index, self.video_version, self.display_time, self.network_time, self.network_ptr, \
					self.yaw_predict_quan, self.yaw_trace[self.video_seg_index*VIDEO_FPS + VIDEO_FPS/2],\
					self.pitch_predict_quan, self.pitch_trace[self.video_seg_index*VIDEO_FPS + VIDEO_FPS/2],\
					temporal_eff, self.rate_cut_version, \
					self.chunk_hit])
				if not COMPARISON:
					assert self.video_seg_index_el > self.display_time
					assert self.video_seg_index_bl > self.display_time

			else:
				print("Unknown video version")
				assert 1 == 0

			# Record BW
			assert self.download_partial == 0
			video_chunk_size = self.rate_cut[self.rate_cut_version][self.video_version] * CHUNK_DURATION
			if len(self.video_bw_history) == 0:
				bw = video_chunk_size/(self.network_time - DELAY)
			else:
				bw = video_chunk_size/(self.network_time - self.video_bw_history[-1][1] - DELAY)
			self.video_bw_history.append([bw, self.network_time, self.video_version, \
							self.rate_cut_version, self.network_trace[self.network_ptr]])
		
		## Current BW time duration is not enough
		else:
			self.download_partial = 1
			self.video_seg_size -= throughput * duration
			self.network_time = np.ceil(self.network_time + 10.**-8)
			self.network_ptr = int(self.network_time)
			rebuf = np.maximum(duration - self.buffer_size_bl, 0.0)
			self.display_time += np.minimum(duration, self.buffer_size_bl)
			if not COMPARISON:
				assert (round(self.display_time * 100)/100).is_integer()

			if round(self.video_seg_index_el - new_temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
				self.buffer_size_el = np.minimum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
			else:
				if round(self.video_seg_index_el - new_temp_video_display_time, 2) != round(self.buffer_size_el, 2) and self.buffer_size_el != 0:
					print("not equal case 4", self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
				self.buffer_size_el = np.maximum(self.buffer_size_el - duration, 0.0)

			self.buffer_size_bl = np.maximum(self.buffer_size_bl - duration, 0.0)

			if self.video_seg_index != int(np.floor(self.display_time)):
				self.video_seg_index_el = np.maximum(self.video_seg_index_el, int(np.floor(self.display_time)) + 1)
			else:
				assert self.video_seg_index_el == int(np.floor(self.display_time))
			self.video_seg_index_bl = np.maximum(self.video_seg_index_bl, int(round(self.display_time * 100)/100))

		# Finish on second, record nencessary information
		if np.floor(round(self.display_time*100000000)/100000000) != np.floor(new_temp_video_display_time):
			# Predict alpha curve and record accuracy 
			# self.display_time = (round(self.display_time*100000000))/100000000
			if self.display_time >= 2.0:
				self.calculate_alpha()
				self.alpha_history.append([self.display_time-1, uti.record_alpha(self.yaw_trace, self.pitch_trace, self.display_time, VIDEO_LEN),\
										self.rate_cut_version])

			if self.buffer_size_bl == 0:
				self.bl_freezing_count += 1
				self.freezing_time += np.round(rebuf, 4)
			if self.buffer_size_el == 0:
				self.el_freezing_count += 1

			self.buffer_history.append([round(self.buffer_size_bl*100)/100, round(self.buffer_size_el*100)/100, self.display_time])
			# print("Current time is %s, and buffer length is %s, %s" %(self.display_time, self.buffer_size_bl, self.buffer_size_el))
			# print("bl and el index is %s and %s" % (self.video_seg_index_bl, self.video_seg_index_el))
		if (self.video_seg_index_el >= VIDEO_LEN and self.video_seg_index_bl >= VIDEO_LEN and self.download_partial == 0):
			# Make up for last chunk info
			print("make up info, time is %s and buffer is: %s" % (self.display_time, self.buffer_size_el))
			final_time_left = np.ceil(self.display_time) - self.display_time
			temp_buffer_size_bl = self.buffer_size_bl - final_time_left
			temp_buffer_size_el = self.buffer_size_el - final_time_left
			self.buffer_history.append([round(temp_buffer_size_bl*100)/100, round(temp_buffer_size_el*100)/100, np.ceil(self.display_time)])

		return temp_video_display_time, recording_el

	def calculate_alpha(self, buffer_range = BUFFER_RANGE):
		# print("calculate alpha time is: %s" %self.display_time)
		for i in range(np.minimum(buffer_range, len(self.alpha_history))):
			# Calculate accuracy for each prediction
			yaw_predict_value = self.alpha_history[-(i+1)][1][i][0]
			yaw_predict_quan = self.alpha_history[-(i+1)][1][i][1]
			pitch_predict_value = self.alpha_history[-(i+1)][1][i][2]
			pitch_predict_quan = self.alpha_history[-(i+1)][1][i][3]

			start_frame = int(self.display_time*VIDEO_FPS) - VIDEO_FPS
			end_frame = int(self.display_time*VIDEO_FPS)
			real_yaw_trace = self.yaw_trace[start_frame:end_frame]
			real_pitch_trace = self.pitch_trace[start_frame:end_frame]
			alpha_value = 0.0
			if not COMPARISON:
				# Using trajectory to calcualte accuracy
				alpha_value = uti.cal_accuracy(yaw_predict_value, yaw_predict_quan, pitch_predict_value, pitch_predict_quan,\
												real_yaw_trace, real_pitch_trace, 1.0, 'quan')
			else:
				# Using Chenge's hitrate trace
				alpha_value = comparison.get_hitrate(self.display_time, self.hitrate, i)

			# print(alpha_value)
			self.alpha_history[-(i+1)][1][i][4] = alpha_value

	def PI_control(self, sniff_bw):
		current_video_version = -1
		video_seg_index = -1
		if self.buffer_size_bl < Q_REF_BL and self.video_seg_index_bl < VIDEO_LEN:
			current_video_version = 0
			video_seg_index = self.video_seg_index_bl

		elif (self.buffer_size_bl >= Q_REF_BL and self.video_seg_index_el < VIDEO_LEN) \
			or (self.video_seg_index_bl >= VIDEO_LEN and self.video_seg_index_el < VIDEO_LEN):
			u_p = KP * (self.buffer_size_el - self.target_et_buffer)
			u_i = 0
			if len(self.buffer_history) != 0:
				for index in range(1, np.minimum(PI_RANGE+1, len(self.buffer_history)+1)):
					u_i += KI * (self.buffer_history[-index][1] - self.target_et_buffer)
			u = u_i + u_p

			v = u + 1
			delta_time = self.buffer_size_el
			R_hat = np.minimum(v, delta_time/CHUNK_DURATION) * sniff_bw
			# print(self.display_time, v, delta_time, sniff_bw, R_hat, self.rate_cut[self.rate_cut_version][3])
			if R_hat >= PI_SMOOTH * self.rate_cut[self.rate_cut_version][3]:
				current_video_version = 3
			elif R_hat >= PI_SMOOTH * self.rate_cut[self.rate_cut_version][2]:
				current_video_version = 2
			else:
				current_video_version = 1
			video_seg_index = self.video_seg_index_el
			# print("el index is %s and version is %s, cause u: %s and delta_time: %s and sniff_bw: %s R_hat; %s" %\
				# (self.video_seg_index_el, current_video_version, u, delta_time, sniff_bw, R_hat))
		self.video_version = current_video_version
		self.video_seg_index = video_seg_index
		# print("going to download: %s at %s" %(self.video_version, self.video_seg_index))
		return
		
	def update_seg_size(self):
		#	based on whole video trace or rate cut
		# self.video_seg_size = self.video_trace[self.video_version][self.video_seg_index]
		self.video_seg_size = self.rate_cut[self.rate_cut_version][self.video_version]
		return 

def main():
	# network_trace = loadNetworkTrace(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	if not COMPARISON:
		half_sec_network_trace, network_trace = load_5G.load_5G_Data(REGULAR_CHANNEL_TRACE, VIDEO_LEN)
	else:
		network_trace = cmp_test.load_single_trace()

	# network_delay = load_5G.load_5G_latency(DELAY_TRACE)
	if REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_5.txt':		#
		average_bw, _ = uti.show_network(network_trace)
		print("above is real value")
		# Whether using whole 450s to calculate, if comment, using 450; need to change alpha curve in utilities.py also, commnet for static 2
		# Disable for all dynamic two tier simulations
		#################### For static 1
		# average_bw, _ = uti.show_network(network_trace[:150])
		####################
	else:
		average_bw, _ = uti.show_network(network_trace)

	if not COMPARISON:
		if not REVISION:
			yaw_trace, pitch_trace = uti.load_viewport(VIEWPORT_TRACE_FILENAME_NEW, VIDEO_LEN)
		else:
			yaw_trace, pitch_trace = uti.load_pickle_viewport(VIEWPORT_TRACE_FILENAME_NEW, VIDEO_LEN, USER)

	else:
		yaw_trace, pitch_trace = comparison.cmp_load_viewport(CMP_VP_TRACE_FILENAME, VIDEO_LEN)

	if not COMPARISON:
		# For JETCAS version, there is control in uti with: FIX_INIT, disable it for STATIC; if do dynamic, use fix init
		if not REVISION:
			init_video_rate, optimal_buffer_length, alpha_idx, gamma_idx = uti.load_init_rates(average_bw, REGULAR_CHANNEL_TRACE, VIEWPORT_TRACE_FILENAME_NEW, CODING_TYPE)
		else:
			init_video_rate, optimal_buffer_length, alpha_idx, gamma_idx = uti.load_init_rates(average_bw, REGULAR_CHANNEL_TRACE, VIEWPORT_TRACE_FILENAME_NEW, CODING_TYPE, user = USER, do_dynamic = DO_DYNAMIC)
	
	# For calculating gamma curve with static buffer
	# Actually the gamma curve is already calculated, and modeled into curves for different ave, std. Using index 
	# init_video_rate, optimal_buffer_length, alpha_idx, gamma_idx = uti.load_init_rates(average_bw, REGULAR_CHANNEL_TRACE, VIEWPORT_TRACE_FILENAME_NEW, CODING_TYPE, False, Q_REF_EL)	
	
	else:
		# For MMSys19 comparison
		init_video_rate, optimal_buffer_length, alpha_idx, gamma_idx = comparison.cmp_load_init_rates(average_bw, REGULAR_CHANNEL_TRACE, CMP_VP_TRACE_FILENAME, CODING_TYPE)


	video_trace = uti.generate_video_trace(init_video_rate, VIDEO_LEN)

	streaming_sim = Streaming(network_trace, yaw_trace, pitch_trace, video_trace, init_video_rate, optimal_buffer_length, alpha_idx, gamma_idx)
		
	streaming_sim.run()

	uti.show_result(streaming_sim, VIDEO_LEN, CODING_TYPE)


def test_all(tp_traces):

	yaw_trace, pitch_trace = comparison.cmp_load_viewport(CMP_VP_TRACE_FILENAME, VIDEO_LEN)
	users = [0, 7, 10, 25, 45]
	# users = [0]
	for user in users:
		hitrate = pickle.load(open(HITRATE_DICT, "rb"))[user]	# For user 0
		# print(len(hitrate))
		rewards = []
		rewards_list = []
		deliver_list = []
		for i in range(len(tp_traces)):
			network_trace = tp_traces[i]
			# print(len(network_trace))
			average_bw = np.mean(network_trace)

			init_video_rate, optimal_buffer_length, alpha_idx, gamma_idx = comparison.cmp_load_init_rates(average_bw, REGULAR_CHANNEL_TRACE, CMP_VP_TRACE_FILENAME, CODING_TYPE)
			video_trace = uti.generate_video_trace(init_video_rate, VIDEO_LEN)

			streaming_sim = Streaming(network_trace, yaw_trace, pitch_trace, video_trace, init_video_rate, optimal_buffer_length, alpha_idx, gamma_idx, hitrate)
				
			streaming_sim.run()

			total_reward, deliver_bitrate = comparison.cmp_show_result(streaming_sim, VIDEO_LEN, CODING_TYPE)
			rewards.append(sum(total_reward))
			rewards_list.append(total_reward)
			deliver_list.append(deliver_bitrate)
			# print total_reward
		# print(np.amax(rewards))
		np.savetxt('cmp_total_rewards_'+ str(user) +'.txt', rewards, fmt='%1.2f')
		# np.savetxt('cmp_list_rewards_'+ str(user) +'.txt', rewards_list[0], fmt='%1.2f')
		# np.savetxt('cmp_list_delivery_'+ str(user) +'.txt', deliver_list[0], fmt='%1.2f')

	return
	
if __name__ == '__main__':
	# Main is not for comparison testing
	main()


