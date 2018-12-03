import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G
import math
import utilities as uti


VIDEO_LEN = 300
VIDEO_FPS = 30

REVISION = 1

REGULAR_CHANNEL_TRACE = './traces/bandwidth/BW_Trace_5G_5.txt'  # 1: partially disturbed  2: unstable  3: stable   4: medium_liyang 5:medium_fanyi

if REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_5.txt':
	VIDEO_LEN = 450
REGULAR_MULTIPLE = 1
REGULAR_ADD = 0

if not REVISION:
	VIEWPORT_TRACE_FILENAME_NEW = './traces/output/Video_9_alpha_beta_new.mat'    ##  9 for 1,  13 for 2, in previous version
	USER = -1
else:
	VIEWPORT_TRACE_FILENAME_NEW = './traces/output/gt_theta_phi_vid_3.p'		# For revision
	USER = 0																	# CHange USER to change FOV trace  0 for 1 (index 2 here),  6 for 2 (index 3 here)


if VIEWPORT_TRACE_FILENAME_NEW == './traces/output/Video_9_alpha_beta_new.mat':
	ALPHA_INDEX = 0
elif VIEWPORT_TRACE_FILENAME_NEW == './traces/output/Video_13_alpha_beta_new.mat':
	ALPHA_INDEX = 1
elif VIEWPORT_TRACE_FILENAME_NEW == './traces/output/gt_theta_phi_vid_3.p':
	if USER == 0:
		ALPHA_INDEX = 2	
	elif USER == 6:
		ALPHA_INDEX = 3

if REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_0.txt':
	GAMMA_INDEX = 0
elif REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_1.txt':
	GAMMA_INDEX = 1
elif REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_2.txt':
	GAMMA_INDEX = 2
elif REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_3.txt':
	GAMMA_INDEX = 3
elif REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_4.txt':
	GAMMA_INDEX = 4
elif REGULAR_CHANNEL_TRACE == './traces/bandwidth/BW_Trace_5G_5.txt':
	GAMMA_INDEX = 5


BUFFER_INIT = 2
Q_REF = 2
BUFFER_THRESH = Q_REF + 2

CHUNK_DURATION = 1.0
KP = 0.45		# P controller
KI = 0.01		# I controller
PI_RANGE = 10
DELAY = 0.01		# second


class Streaming(object):
	def __init__(self, network_trace, yaw_trace, pitch_trace, video_rate):
		
		self.dy_type = 'fix'
		self.fov_file = ALPHA_INDEX
		self.network_file = GAMMA_INDEX


		self.network_trace = network_trace
		self.yaw_trace = yaw_trace
		self.pitch_trace = pitch_trace
		self.download_partial = 0

		self.rates = video_rate
		self.video_chunk_size = 0.0

		self.network_ptr = 0
		self.network_time = 0.0
		# self.display_time = 0.0

		self.video_version = 0
		self.video_seg_index = BUFFER_INIT
		self.buffer_size = BUFFER_INIT
		self.buffer_history = []

		self.video_bw_history = []
		self.evr_recordset = []
		self.freezing_time = 0.0

		self.yaw_predict_value = 0.0
		self.yaw_predict_quan = 0
		self.pitch_predict_value = 0.0
		self.pitch_predict_quan = 0


	def run(self):
		data_sent = 0.0
		while self.video_seg_index < VIDEO_LEN:
			if not self.download_partial:
				sniff_bw = uti.predict_bw(self.video_bw_history)
				# self.bw_info = np.append(self.bw_info, [sniff_bw, self.network_time])
				self.PI_control(sniff_bw)

				self.yaw_predict_value, self.yaw_predict_quan = uti.predict_yaw_trun(self.yaw_trace, self.network_time, self.video_seg_index)
				self.pitch_predict_value, self.pitch_predict_quan = uti.predict_pitch_trun(self.pitch_trace, self.network_time, self.video_seg_index)
				# print(self.yaw_predict_value, self.yaw_trace[self.video_seg_index*VIDEO_FPS + 15])
			data_sent = self.fetching(data_sent)
			if not self.download_partial:
				assert data_sent == 0.0
			else:
				assert data_sent > 0.0

	def fetching(self, sent):
		throughput = self.network_trace[self.network_ptr]
		duration = self.network_ptr + 1.0 - self.network_time
		temp_time = self.network_time
		if not self.download_partial:
			if duration > DELAY:
				duration -= DELAY
				self.network_time += DELAY
				if round(self.video_seg_index - temp_time , 3) > round(self.buffer_size, 3) and self.buffer_size != 0:
					self.buffer_size = np.minimum(self.buffer_size, np.maximum(self.video_seg_index - self.network_time, 0.0))
				else:
					if round(self.video_seg_index - temp_time, 3) != round(self.buffer_size, 3) and self.buffer_size != 0:
						print("not equal case 1", self.video_seg_index, temp_time, self.buffer_size)
					self.buffer_size = np.maximum(self.buffer_size - DELAY, 0.0)				
			else:
				if self.video_seg_index == self.network_ptr + 1:
					# Insert zero for current time
					# And request next chunk
					self.evr_recordset.appned([self.video_seg_index, -1, self.network_time, -1, -1, -1, -1])
					self.video_seg_index += 1
					if self.video_seg_index < VIDEO_LEN:
						self.yaw_predict_value, self.yaw_predict_quan = uti.predict_yaw_trun(self.yaw_trace, self.network_time, self.video_seg_index)
						self.pitch_predict_value, self.pitch_predict_quan = uti.predict_pitch_trun(self.pitch_trace, self.network_time, self.video_seg_index)
						# print(self.yaw_predict_value, self.yaw_trace[self.video_seg_index*VIDEO_FPS + 15])
						sniff_bw = uti.predict_bw(self.video_bw_history)
						self.PI_control(sniff_bw)
					else:
						return 0.0

				temp_delay = DELAY - duration
				self.network_time = self.network_ptr + 1.0
				self.network_ptr += 1
				if round(self.video_seg_index - temp_time , 3) > round(self.buffer_size, 3) and self.buffer_size != 0:
					self.buffer_size = np.minimum(self.buffer_size, np.maximum(self.video_seg_index - self.network_time, 0.0))
				else:
					if round(self.video_seg_index - temp_time, 3) != round(self.buffer_size, 3) and self.buffer_size != 0:
						print("not equal case 1", self.video_seg_index, temp_time, self.buffer_size)
					self.buffer_size = np.maximum(self.buffer_size - duration, 0.0)				
				# self.buffer_history.append([self.buffer_size, self.network_time])
				if not round(self.buffer_size, 3).is_integer():
					print("Not integer: %s" % round(self.buffer_size, 3))
				temp_time = self.network_time
				# assert self.buffer_size == 0.0
				if self.network_ptr > len(self.network_trace):
					print("network trace is not enough, case 0")
					self.network_ptr = 0
					self.network_time = 0.0
				self.network_time += temp_delay
				if round(self.video_seg_index - temp_time , 3) > round(self.buffer_size, 3) and self.buffer_size != 0:
					self.buffer_size = np.minimum(self.buffer_size, np.maximum(self.video_seg_index - self.network_time, 0.0))
				else:
					if round(self.video_seg_index - temp_time, 3) != round(self.buffer_size, 3) and self.buffer_size != 0:
						print("not equal case 1", self.video_seg_index, temp_time, self.buffer_size)
					self.buffer_size = np.maximum(self.buffer_size - duration, 0.0)
				duration = self.network_ptr + 1.0 - self.network_time
				throughput = self.network_trace[self.network_ptr]

		temp_time = self.network_time
		payload = throughput * duration
		if payload + sent > self.video_chunk_size:
			self.download_partial = 0
			assert self.video_seg_index > self.network_ptr
			fraction_time = (self.video_chunk_size - sent)/throughput
			self.network_time += fraction_time
			assert self.network_time < self.network_ptr + 1

			if round(self.video_seg_index - temp_time , 3) > round(self.buffer_size, 3) and self.buffer_size != 0:
				self.buffer_size = np.minimum(self.buffer_size, np.maximum(self.video_seg_index - self.network_time, 0.0))
			else:
				if round(self.video_seg_index - temp_time, 3) != round(self.buffer_size, 3) and self.buffer_size != 0:
					print("not equal case 2", self.video_seg_index, temp_time, self.buffer_size)
				self.buffer_size = np.maximum(self.buffer_size - fraction_time, 0.0)

			self.buffer_size += CHUNK_DURATION
			if len(self.video_bw_history) == 0:
				bw = self.video_chunk_size/(self.network_time - DELAY)
			else:
				bw = self.video_chunk_size/(self.network_time - self.video_bw_history[-1][1] - DELAY)
			self.video_bw_history.append([bw, self.network_time])
			self.evr_recordset.append([self.video_seg_index, self.video_version, self.network_time, \
					self.yaw_predict_quan, self.yaw_trace[self.video_seg_index*VIDEO_FPS + VIDEO_FPS/2],\
					self.pitch_predict_quan, self.pitch_trace[self.video_seg_index*VIDEO_FPS + VIDEO_FPS/2]])
			self.video_seg_index += 1

			if round(self.buffer_size, 3) > BUFFER_THRESH:
				# print("enter sleep, time is %s, ptr is %s, and buffer is %s, seg is %s" %(\
				# 					self.network_time, self.network_ptr, self.buffer_size, self.video_seg_index))
				index_gap = self.video_seg_index - self.network_time
				assert index_gap >= self.buffer_size - BUFFER_THRESH
				sleep_time = np.maximum(index_gap - BUFFER_THRESH, self.buffer_size - BUFFER_THRESH)
				print("sleep_time is %s" % sleep_time)
				if sleep_time > CHUNK_DURATION:
					assert index_gap > self.buffer_size - BUFFER_THRESH
					assert round(self.buffer_size, 3).is_integer()
					first_sleep = sleep_time - np.floor(sleep_time)
					sleep_time -= first_sleep
					self.network_time += first_sleep
					assert round(self.network_time, 3).is_integer()
					self.network_ptr += 1
					# self.buffer_history.append([self.buffer_size, self.network_time])
				self.network_time += sleep_time
				if not round(self.network_time, 3).is_integer():
					print(self.network_time)
				assert round(self.network_time, 3).is_integer()
				self.network_ptr += 1
				self.buffer_size -= sleep_time
				if not round(self.buffer_size, 3).is_integer():
					print("special case, buffer size is not integer after sleep: %s" % round(self.buffer_size, 3))
				self.buffer_history.append([self.buffer_size, self.network_time])
			# print("after downloading, time is %s, ptr is %s, and buffer is %s, seg is %s" %(\
			# 						self.network_time, self.network_ptr, self.buffer_size, self.video_seg_index))
			return 0.0
		else:
			self.download_partial = 1
			self.network_time += duration
			assert round(self.network_time,3).is_integer()
			self.network_ptr += 1
			if round(self.video_seg_index - temp_time , 3) > round(self.buffer_size, 3) and self.buffer_size != 0:
				self.buffer_size = np.minimum(self.buffer_size, np.maximum(self.video_seg_index - self.network_time, 0.0))
			else:
				if round(self.video_seg_index - temp_time, 3) != round(self.buffer_size, 3) and self.buffer_size != 0:
					print("not equal case 3", self.video_seg_index, temp_time, self.buffer_size)
				self.buffer_size = np.maximum(self.buffer_size - duration, 0.0)


			self.buffer_history.append([self.buffer_size, self.network_time])
			if self.video_seg_index == self.network_ptr:
				self.evr_recordset.append([self.video_seg_index, -1, self.network_time, -1, -1, -1, -1])				
				self.video_seg_index = self.network_ptr + 1
				self.download_partial = 0
			# print("not finish, time is %s, ptr is %s, and buffer is %s, seg is %s" %(self.network_time, \
			# 							self.network_ptr, self.buffer_size, self.video_seg_index))
				return 0.0
			else:
				return payload 

	def PI_control(self, sniff_bw):
		u_p = KP * (self.buffer_size - Q_REF)
		u_i = 0

		if len(self.buffer_history) != 0:
			for index in range(1, np.minimum(PI_RANGE+1, len(self.buffer_history)+1)):
				u_i += KI * (self.buffer_history[-index][0] - Q_REF)
		u = u_i + u_p

		v = u + 1
		delta_time = self.buffer_size
		R_hat = np.minimum(v, delta_time/CHUNK_DURATION) * sniff_bw
		# R_hat = v*sniff_bw

		if R_hat < self.rates[0]:
			current_video_version = 0
		else:
			for i in reversed(range(len(self.rates))):
				if R_hat >= self.rates[i]:
					current_video_version = i
					break
			assert not current_video_version == -1

		# if R_hat >= self.rates[5]:
		# 	current_video_version = 5
		# elif R_hat >= self.rates[4]:
		# 	current_video_version = 4
		# elif R_hat >= self.rates[3]:
		# 	current_video_version = 3
		# elif R_hat >= self.rates[2]:
		# 	current_video_version = 2
		# elif R_hat >= self.rates[1]:
		# 	current_video_version = 1
		# else:
		# 	current_video_version = 0

		self.video_version = current_video_version
		self.video_chunk_size = self.rates[current_video_version]
		return 

def main():
	half_sec_network_trace, network_trace = load_5G.load_5G_Data(REGULAR_CHANNEL_TRACE, VIDEO_LEN, REGULAR_MULTIPLE, REGULAR_ADD)
	average_bw = uti.show_network(network_trace)
	if not REVISION:
		yaw_trace, pitch_trace = uti.load_viewport(VIEWPORT_TRACE_FILENAME_NEW, VIDEO_LEN)
	else:
		yaw_trace, pitch_trace = uti.load_pickle_viewport(VIEWPORT_TRACE_FILENAME_NEW, VIDEO_LEN, USER)

	video_rate = uti.generate_fov_rate()

	streaming_sim = Streaming(network_trace, yaw_trace, pitch_trace, video_rate)

	streaming_sim.run()

	uti.show_fov_result(streaming_sim, VIDEO_LEN, BUFFER_INIT)


if __name__ == '__main__':
	main()