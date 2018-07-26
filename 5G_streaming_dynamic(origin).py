# Prepare for dynamic, original version, same function with 5g streaming
# No dynamic in this file


import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G
import math
import utilities as uti


VIDEO_LEN = 300
VIDEO_FPS = 30
# BW trace
REGULAR_CHANNEL_TRACE = './traces/bandwidth/BW_Trace_5G_2.txt'  # 1: partially disturbed  2: unstable  3: stable   4: medium_liyang 5:medium_fanyi
# DELAY_TRACE = 'delay_1.txt'
REGULAR_MULTIPLE = 1
REGULAR_ADD = 0
# VP trace
VIEWPORT_TRACE_FILENAME_NEW = './traces/output/Video_9_alpha_beta_new.mat'    ##  9 for 1,  13 for 2

# System parameters
BUFFER_BL_INIT = 10
BUFFER_EL_INIT = 1
Q_REF_BL = 10
Q_REF_EL = 1
ET_MAX_PRED = Q_REF_EL + 1

# Chunk parameters
CHUNK_DURATION = 1.0

#Others
KP = 0.6		# P controller
KI = 0.01		# I controller
PI_RANGE = 30
DELAY = 0.02		# second

BW_DALAY_RATIO = 0.95

class Streaming(object):
	def __init__(self, network_trace, yaw_trace, pitch_trace, video_trace, rate_cut):
		self.network_trace = network_trace
		self.yaw_trace = yaw_trace
		self.pitch_trace = pitch_trace
		self.video_trace = video_trace
		self.rate_cut = []
		self.rate_cut.append(rate_cut)
		self.rate_cut_version = 0

		self.network_ptr = 0
		self.network_time = 0.0
		self.display_time = 0.0

		self.video_seg_index_bl = BUFFER_BL_INIT 
		self.video_seg_index_el = BUFFER_EL_INIT
		self.buffer_size_bl = BUFFER_BL_INIT
		self.buffer_size_el = BUFFER_EL_INIT
		self.buffer_history = []

		self.download_partial = 0
		self.video_seg_size = 0.0

		# self.bw_info = []
		self.video_bw_history = []
		self.video_version = 0
		self.video_seg_index = 0

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

	def run(self):
		while self.video_seg_index_bl < VIDEO_LEN or \
			(self.video_seg_index_bl >= VIDEO_LEN and self.video_seg_index_el < VIDEO_LEN):
			if not self.download_partial:
				sniff_bw = uti.predict_bw(self.video_bw_history)
				# self.bw_info = np.append(self.bw_info, [sniff_bw, self.network_time])
				self.PI_control(sniff_bw)
				self.update_seg_size()
			
				temp_index = self.video_seg_index
				if self.video_version != 0:
					assert self.video_version != -1
					self.yaw_predict_value, self.yaw_predict_quan = uti.predict_yaw_trun(self.yaw_trace, self.display_time, self.video_seg_index)
					self.pitch_predict_value, self.pitch_predict_quan = uti.predict_pitch_trun(self.pitch_trace, self.display_time, self.video_seg_index)
			previous_time, recording_el = self.fetching()

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
		if self.video_version >= 1 and round(self.buffer_size_el, 3) > ET_MAX_PRED:
			assert self.download_partial != 1
				# print("Current tiem is %s, and buffer length is %s, %s, is downloading %s" %(self.display_time, self.buffer_size_bl, self.buffer_size_el, self.video_version))
			# need to sleep, nothing should be recorded
			recording_el = 0
			#Calculate sleep time
			sleep_time = self.video_seg_index_el - ET_MAX_PRED - temp_video_display_time
			assert sleep_time <= 1
			self.display_time += sleep_time 
			self.network_time += sleep_time
			self.network_ptr = int(np.floor(self.network_time))

			if round(self.display_time, 2) != round(self.network_time, 2):
				print("Network time and display tims is not sync")
			if round(self.video_seg_index_el - ET_MAX_PRED, 2) != round(self.display_time, 2):
				print("Two methods for sleep are not sync")

			if round(self.video_seg_index_el - temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
				self.buffer_size_el = np.maximum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
			else:
				if round(self.video_seg_index_el - temp_video_display_time, 2) != round(self.buffer_size_el, 2) and self.buffer_size_el != 0:
					print(self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
				self.buffer_size_el = np.maximum(self.buffer_size_el - sleep_time, 0.0)
			if not round(self.network_time, 3).is_integer():
				print("Current network time is %s, not integer" % self.network_time)
			self.buffer_size_bl -= sleep_time
			self.video_bw_history.append([self.network_trace[self.network_ptr]*BW_DALAY_RATIO, self.network_time, -1, \
									self.rate_cut_version, self.network_trace[self.network_ptr]])
			print("after sleep, Current tiem is %s, and buffer length is %s, %s, is downloading %s" %(self.display_time, self.buffer_size_bl, self.buffer_size_el, self.video_version))
			self.buffer_history.append([round(round(self.buffer_size_bl*100 + 2)/100), round(round(self.buffer_size_el*100 + 2)/100), self.display_time])
			return temp_video_display_time, recording_el

		# All below don't need to consider sleep!
		if not self.download_partial:
			current_time_left = np.ceil(temp_video_display_time + 10.**-8) - temp_video_display_time
			
			# For the delay part, need record
			if current_time_left < DELAY:
				print("Delay larger than current time left, tiem left is %s" % current_time_left)
				self.buffer_history.append([round(round(self.buffer_size_bl*100 + 2)/100), round(round(self.buffer_size_el*100 + 2)/100), self.display_time])

			self.network_time += DELAY
			self.network_ptr = int(self.network_time)

			# No freezing
			if self.buffer_size_bl >= DELAY:
				self.display_time += DELAY
				self.buffer_size_bl -= DELAY
				if round(self.video_seg_index_el - temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
					self.buffer_size_el = np.maximum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
				else:
					if round(self.video_seg_index_el - temp_video_display_time, 2) != round(self.buffer_size_el, 2):
						print(self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
					self.buffer_size_el = np.maximum(self.buffer_size_el - DELAY, 0.0)
			else:
				assert self.buffer_size_el == 0
				print("will not happen in this simulation!!!!!!!!!!!!!!!!!!!!!")

			## Time after request
			new_temp_video_display_time = self.display_time

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
			if round(self.video_seg_index_el - new_temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
				self.buffer_size_el = np.minimum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
			else:
				if round(self.video_seg_index_el - new_temp_video_display_time, 2) != round(self.buffer_size_el, 2):
					print(self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
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
				assert self.buffer_size_el <= ET_MAX_PRED	
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
					temporal_eff, self.rate_cut_version])
				assert self.video_seg_index_el > self.display_time
				assert self.video_seg_index_bl > self.display_time

			else:
				print("Unknown video version")
				assert 1 == 0

			# Record BW
			assert self.download_partial == 0
			video_chunk_size = self.rate_cut[self.rate_cut_version][self.video_version] * CHUNK_DURATION
			if len(self.video_bw_history) == 0:
				bw = video_chunk_size/self.network_time
			else:
				bw = video_chunk_size/(self.network_time - self.video_bw_history[-1][1])
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
			assert (round(self.display_time * 100)/100).is_integer()

			if round(self.video_seg_index_el - new_temp_video_display_time, 3) > round(self.buffer_size_el, 3) and self.buffer_size_el != 0:
				self.buffer_size_el = np.minimum(self.buffer_size_el, np.maximum(self.video_seg_index_el - self.display_time, 0.0))
			else:
				if round(self.video_seg_index_el - new_temp_video_display_time, 2) != round(self.buffer_size_el, 2):
					print(self.video_seg_index_el, temp_video_display_time, self.buffer_size_el)
				self.buffer_size_el = np.maximum(self.buffer_size_el - duration, 0.0)

			self.buffer_size_bl = np.maximum(self.buffer_size_bl - duration, 0.0)

			self.video_seg_index_el = np.maximum(self.video_seg_index_el, int(np.floor(self.display_time)) + 1)
			self.video_seg_index_bl = np.maximum(self.video_seg_index_bl, int(round(self.display_time * 100)/100))

		if np.floor(round(self.display_time*100000)/100000) != np.floor(new_temp_video_display_time):
			if self.buffer_size_bl == 0:
				self.bl_freezing_count += 1
				self.freezing_time += rebuf
			if self.buffer_size_el == 0:
				self.el_freezing_count += 1

			self.buffer_history.append([round(self.buffer_size_bl*100)/100, round(self.buffer_size_el*100)/100, self.display_time])
			print("Current tiem is %s, and buffer length is %s, %s" %(self.display_time, self.buffer_size_bl, self.buffer_size_el))
			print("bl and el index is %s and %s" % (self.video_seg_index_bl, self.video_seg_index_el))
		if (self.video_seg_index_el >= VIDEO_LEN and self.video_seg_index_bl >= VIDEO_LEN):
			# Make up for last chunk info
			final_time_left = np.ceil(self.display_time) - self.display_time
			temp_buffer_size_bl = self.buffer_size_bl - final_time_left
			temp_buffer_size_el = self.buffer_size_el - final_time_left
			self.buffer_history.append([round(temp_buffer_size_bl*100)/100, round(temp_buffer_size_el*100)/100, np.ceil(self.display_time)])

		return temp_video_display_time, recording_el

	def PI_control(self, sniff_bw):
		current_video_version = -1
		video_seg_index = -1
		if self.buffer_size_bl < Q_REF_BL and self.video_seg_index_bl < VIDEO_LEN:
			current_video_version = 0
			video_seg_index = self.video_seg_index_bl

		elif (self.buffer_size_bl >= Q_REF_BL and self.video_seg_index_el < VIDEO_LEN) \
			or (self.video_seg_index_bl >= VIDEO_LEN and self.video_seg_index_el < VIDEO_LEN):
			u_p = KP * (self.buffer_size_el - Q_REF_EL)
			u_i = 0
			if len(self.buffer_history) != 0:
				for index in range(1, min(PI_RANGE+1, len(self.buffer_history)+1)):
					u_i += KI * (self.buffer_history[-index][1] - Q_REF_EL)
			u = u_i + u_p

			v = u + 1
			delta_time = self.buffer_size_el
			R_hat = np.minimum(v, delta_time/CHUNK_DURATION) * sniff_bw

			if R_hat >= self.rate_cut[self.rate_cut_version][3]:
				current_video_version = 3
			elif R_hat >= self.rate_cut[self.rate_cut_version][2]:
				current_video_version = 2
			else:
				current_video_version = 1
			video_seg_index = self.video_seg_index_el
		self.video_version = current_video_version
		self.video_seg_index = video_seg_index
		print("going to download: %s at %s" %(self.video_version, self.video_seg_index))
		return
		
	def update_seg_size(self):
		#	based on whole video trace or rate cut
		# self.video_seg_size = self.video_trace[self.video_version][self.video_seg_index]
		self.video_seg_size = self.rate_cut[self.rate_cut_version][self.video_version]
		return

def main():
	# network_trace = loadNetworkTrace(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	half_sec_network_trace, network_trace = load_5G.load_5G_Data(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	# network_delay = load_5G.load_5G_latency(DELAY_TRACE)
	average_bw = uti.show_network(network_trace)
	yaw_trace, pitch_trace = uti.load_viewport(VIEWPORT_TRACE_FILENAME_NEW)

	init_video_rate = uti.load_init_rates(average_bw)
	video_trace = uti.generate_video_trace(init_video_rate)

	streaming_sim = Streaming(network_trace, yaw_trace, pitch_trace, video_trace, init_video_rate)
	
	streaming_sim.run()

	uti.show_result(streaming_sim)


if __name__ == '__main__':
	main()