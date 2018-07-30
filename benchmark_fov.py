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

VIEWPORT_TRACE_FILENAME_NEW = './traces/output/Video_9_alpha_beta_new.mat'    ##  9 for 1,  13 for 2

BUFFER_INIT = 1
Q_REF = 1
BUFFER_THRESH = Q_REF + 1

CHUNK_DURATION = 1.0
KP = 0.6		# P controller
KI = 0.01		# I controller
PI_RANGE = 10
DELAY = 0.01		# second




def main():
	half_sec_network_trace, network_trace = load_5G.load_5G_Data(REGULAR_CHANNEL_TRACE, REGULAR_MULTIPLE, REGULAR_ADD)
	average_bw = uti.show_network(network_trace)

	video_rate = uti.generate_360_rate()

	streaming_sim = Streaming(network_trace, video_rate)

	streaming_sim.run()


if __name__ == '__main__':
	main()