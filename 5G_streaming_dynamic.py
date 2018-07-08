import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G
import math
import utilities as uti


VIDEO_LEN = 300
# BW trace
REGULAR_CHANNEL_TRACE = 'BW_Trace_5G_1.txt'  # 1: partially disturbed  2: unstable  3: stable   4: medium_liyang 5:medium_fanyi
DELAY_TRACE = 'delay_1.txt'
REGULAR_MULTIPLE = 1
REGULAR_ADD = 0
# VP trace
VIEWPORT_TRACE_FILENAME_NEW = 'Video_9_alpha_beta_new.mat'    ##  9 for 1,  13 for 2

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
PI_RANGE = 30
DELAY = 0.02		# second

class Streaming(object):
	def __init__(self, network_trace, yaw_trace, pitch_trace, video_trace, rate_cut):
		self.network_trace = network_trace
		self.yaw_trace = yaw_trace
		self.pitch_trace = pitch_trace
		self.video_trace = video_trace
		self.rate_cut = rate_cut

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

		self.video_bw_history = []
		self.bw_info = []
		self.video_version = 0
		self.video_seg_index = 0

		self.yaw_predict_value = 0.0
		self.yaw_predict_quan = 0
		self.pitch_predict_value = 0.0
		self.pitch_predict_quan = 0

		self.record_info = []

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
			if not self.download_partial and 


	def fetching(self):
		temp_video_display_time = self.display_time
		video_rate = 0
		recording_el = 0
		is_same_el = 0
		record_bw = 1

		if not self.download_partial:
			current_time_left = np.ceil(temp_video_display_time) - temp_video_display_time
			
			# For the delay part, need record
			if current_time_left < DELAY:

			self.display_time += DELAY
			self.network_time += DELAY
			self.network_ptr = int(self.network_time)

		throughput = self.network_trace[self.network_ptr]
		duration = np.ceil(self.network_time) - self.network_time
		if throughput * duration >= self.video_seg_size:
			download_duration =  self.video_seg_size/throughput
			self.download_partial = 0
			if self.video_version == 0:
				self.evr_bl_recordset.append([self.video_seg_index_bl, self.video_version, self.network_time, self.network_ptr])
				self.network_time += download_duration
				self.network_ptr = int(self.network_time)
				self.video_seg_index_bl += 1
				self.video_seg_index_el = np.maximum(int(np.floor(self.display_time))+1, self.video_seg_index_el)



	def PI_control(self, sniff_bw):
		current_video_version = -1
		video_seg_index = -1
		if self.buffer_size_bl < Q_REF_BL and self.video_seg_index_bl < VIDEO_LEN:
			current_video_version = 0
			video_seg_index = self.video_seg_index_bl

		elif (self.buffer_size_bl >= Q_REF_BL and self.video_seg_index_el < VIDEO_LEN) \
			or (self.buffer_size_bl >= VIDEO_LEN and self.video_seg_index_el < VIDEO_LEN):
			u_p = KP * (self.buffer_size_el - Q_REF_EL)
			u_i = 0
			if len(self.buffer_history) != 0:
				for index in range(1, min(PI_RANGE+1, len(self.buffer_history)+1)):
					u_i += KI * (self.buffer_history[-index][1] - Q_REF_EL)
			u = u_i + u_p

			v = u + 1
			delta_time = self.buffer_size_el
			R_hat = np.minimum(v, delta_time/CHUNK_DURATION)

			if R_hat >= self.rate_cut[3]:
				current_video_version = 3
			elif R_hat >= rate_cut[2]:
				current_video_version = 2
			else:
				current_video_version = 1
			video_seg_index = self.video_seg_index_el
		self.video_version = current_video_version
		self.video_seg_index = video_seg_index
		return
		
	def update_seg_size(self):
		self.video_seg_size = self.video_trace[self.video_version][self.video_seg_index]
		return

def main():
	# network_trace = loadNetworkTrace(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	half_sec_network_trace, network_trace = load_5G.load_5G_Data(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	network_delay = load_5G.load_5G_latency(DELAY_TRACE)
	average_bw = uti.show_network(network_trace, network_delay)
	yaw_trace, pitch_trace = uti.load_viewport(VIEWPORT_TRACE_FILENAME_NEW)

	init_video_rate = uti.load_init_rates(average_bw)
	video_trace = uti.generate_video_trace(init_video_rate)

	streaming_sim = Streaming(network_trace, yaw_trace, pitch_trace, video_trace)
	streaming_sim.run()

if __name__ == '__main__':
	main()