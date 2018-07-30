import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import load_5G
import math
import utilities as uti

VIDEO_LEN = 300
REGULAR_CHANNEL_TRACE = './traces/bandwidth/BW_Trace_5G_4.txt'  # 1: partially disturbed  2: unstable  3: stable   4: medium_liyang 5:medium_fanyi
REGULAR_MULTIPLE = 1
REGULAR_ADD = 0

# VIEWPORT_TRACE_FILENAME_NEW = './traces/output/Video_9_alpha_beta_new.mat'    ##  9 for 1,  13 for 2

BUFFER_INIT = 10
Q_REF = 10
BUFFER_THRESH = 20

CHUNK_DURATION = 1.0

KP = 0.6		# P controller
KI = 0.01		# I controller
PI_RANGE = 10
DELAY = 0.01		# second


class Streaming(object):
	def __init__(self, network_trace, video_rate):
		self.network_trace = network_trace
		# self.download_partial = 0

		self.rates = video_rate
		self.video_chunk_size = 0.0

		self.network_ptr = 0
		self.network_time = 0.0
		self.display_time = 0.0

		self.video_seg_index = BUFFER_INIT
		self.buffer_size = BUFFER_INIT
		self.buffer_history = []

		self.video_bw_history = []
		self.evr_recordset = []
		self.freezing_time = 0.0


	def run(self):
		while self.video_seg_index < VIDEO_LEN:
			# if not self.download_partial:
			sniff_bw = uti.predict_bw(self.video_bw_history)
			self.PI_control(sniff_bw)

			delay = 0.0
			video_sent = 0.0
			count_delay = 0
			while True:
				throughput = self.network_trace[self.network_ptr]
				duration = self.network_ptr + 1.0 - self.network_time
				if count_delay == 0:
					if duration > DELAY:
						duration -= DELAY
						self.network_time += DELAY
					else:
						temp_delay = DELAY - duration
						self.network_time = self.network_ptr + 1.0
						self.network_ptr += 1
						if self.network_ptr > len(self.network_trace):
							print("network trace is not enough, case 0")
							self.network_ptr = 0
							self.network_time = 0.0
						self.network_time += temp_delay
						duration = self.network_ptr + 1.0 - self.network_time
					count_delay = 1

				payload = throughput * duration
				if payload + video_sent >= self.video_chunk_size:
					fraction_time = self.video_chunk_size/throughput
					delay += fraction_time
					print("before network time is: %s and ptr is %s" %(self.network_time, self.network_ptr))
					self.network_time += fraction_time
					if self.network_ptr + 1.0 < self.network_time:
						print("network time is: %s and ptr is %s" %(self.network_time, self.network_ptr))
					break

				video_sent += payload
				delay += duration
				self.network_time = self.network_ptr + 1.0
				self.network_ptr += 1

				if self.network_ptr > len(self.network_trace):
					print("network trace is not enough, case 1")
					self.network_ptr = 0
					self.network_time = 0.0

			delay += DELAY					
			# print("delay is %s" % delay)
			rebuf = np.maximum(delay - self.buffer_size, 0.0)
			self.display_time += np.minimum(self.buffer_size, delay)
			self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)
			self.buffer_size += CHUNK_DURATION
			if len(self.video_bw_history) == 0:
				bw = self.video_chunk_size/self.network_time
			else:
				bw = self.video_chunk_size/(self.network_time-self.video_bw_history[-1][1])
			self.video_bw_history.append([bw, self.network_time])
			print("before sleep, network time is %s,%s display time is %s and buffer is %s" %(self.network_time,\
					self.network_ptr, self.display_time, self.buffer_size))

			# sleep case
			if self.buffer_size > BUFFER_THRESH:
				sleep_time = self.buffer_size - BUFFER_THRESH
				assert sleep_time <= CHUNK_DURATION
				self.buffer_size -= sleep_time
				self.display_time += sleep_time
				assert round(self.display_time, 3).is_integer()
				assert round(self.buffer_size, 3).is_integer()
				while True:
					duration = self.network_ptr + 1.0 - self.network_time
					if duration > sleep_time:
						self.network_time += sleep_time
						self.video_bw_history.append([self.network_trace[self.network_ptr], self.network_time])
						break
					sleep_time -= duration
					self.network_time += duration
					self.video_bw_history.append([self.network_trace[self.network_ptr], self.network_time])
					self.network_ptr += 1

					if self.network_ptr > len(self.network_trace):
						print("network trace is not enough, case 2")
						self.network_ptr = 0
						self.network_time = 0.0
			self.evr_recordset.append([self.video_seg_index, self.video_version, rebuf, self.display_time])
			self.buffer_history.append([self.buffer_size, self.display_time])
			self.video_seg_index += 1
			count_delay = 0
			rebuf = 0.0
			print("one round end, network time is %s,%s display time is %s and buffer is %s" %(self.network_time,\
					self.network_ptr, self.display_time, self.buffer_size))
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

		if R_hat >= self.rates[5]:
			current_video_version = 5
		elif R_hat >= self.rates[4]:
			current_video_version = 4
		elif R_hat >= self.rates[3]:
			current_video_version = 3
		elif R_hat >= self.rates[2]:
			current_video_version = 2
		elif R_hat >= self.rates[1]:
			current_video_version = 1
		else:
			current_video_version = 0

		self.video_version = current_video_version
		self.video_chunk_size = self.rates[current_video_version]
		return 


def main():
	half_sec_network_trace, network_trace = load_5G.load_5G_Data(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	average_bw = uti.show_network(network_trace)

	video_rate = uti.generate_360_rate()

	streaming_sim = Streaming(network_trace, video_rate)

	streaming_sim.run()


if __name__ == '__main__':
	main()