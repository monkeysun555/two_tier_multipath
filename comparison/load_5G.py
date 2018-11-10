import scipy.io as sio
import numpy as np
import utilities as uti

# VIDEO_LEN = 300
# NETWORK_TRACE_LEN = VIDEO_LEN 		# For 5G python

FILE_LIST = ['./traces/bandwidth/BW_Trace_5G_0.txt','./traces/bandwidth/BW_Trace_5G_1.txt'\
					,'./traces/bandwidth/BW_Trace_5G_2.txt','./traces/bandwidth/BW_Trace_5G_3.txt'\
					,'./traces/bandwidth/BW_Trace_5G_4.txt', './traces/bandwidth/BW_Trace_5G_5.txt']
# NETWORK_TRACE_LEN = VIDEO_LEN + 100 		# For two-tier multipath and vp only
def load_5G_Data(fname, video_length, addition_length = 0, multiple=1, addition=0 ):
	if fname == './traces/bandwidth/BW_Trace_5G_1.txt':
		multiple = 1.7
		addition = -520
	elif fname == './traces/bandwidth/BW_Trace_5G_2.txt':
		multiple = 0.8
		addition = 80

	if fname == './traces/bandwidth/BW_Trace_5G_5.txt':
		fname1 = './traces/bandwidth/BW_Trace_5G_0.txt'
		with open(fname1) as f:
			content1 = f.readlines()
		content1 = [float(x.strip()) for x in content1]
		content1 = content1[: 2*video_length/3]
		fname2 = './traces/bandwidth/BW_Trace_5G_2.txt'
		with open(fname2) as f:
			content2 = f.readlines()
			multiple = 0.8
			addition = 80
		content2 = [multiple * float(x.strip()) + addition for x in content2]
		content2 = content2[: 2*video_length/3]
		content = np.concatenate((content2, content1), axis=0)

		fname3 = './traces/bandwidth/BW_Trace_5G_4.txt'
		with open(fname3) as f:
			content3 = f.readlines()
		content3 = [float(x.strip()) for x in content3]
		content3 = content3[:2*(video_length/3+addition_length)]
		content = np.concatenate((content, content3), axis=0)

	else:
		with open(fname) as f:
			content = f.readlines()
		# you may also want to remove whitespace characters like `\n` at the end of each line
		content = [max(multiple * float(x.strip()) + addition, -(multiple * float(x.strip()) + addition)) for x in content]
	# print(content, len(content))
	new_content = []
	for i in range(0,video_length+addition_length):
		new_content.append((content[2*i]+content[2*i+1])/2)
	return content[:2*(video_length+addition_length)+1], new_content

def load_5G_latency(fname, video_length=300):
	with open(fname) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [float(x.strip()) for x in content]
	# print(content, len(content))
	return content[:video_length]


def main(fnames):
	for fname in fnames:
		print(fname)
		if fname == './traces/bandwidth/BW_Trace_5G_2.txt':
			half_sec_network_trace, network_trace = load_5G_Data(fname, 300, 0.8, 80)
		elif fname == './traces/bandwidth/BW_Trace_5G_1.txt':
			half_sec_network_trace, network_trace = load_5G_Data(fname, 300, 1.7, -520)
		elif fname == './traces/bandwidth/BW_Trace_5G_5.txt':
			half_sec_network_trace, network_trace = load_5G_Data(fname, 900, 0.8, 80)
		else:
			half_sec_network_trace, network_trace = load_5G_Data(fname, 300)
		average_bw = uti.show_network(network_trace)
		print("<====================>")
	# yaw_trace, pitch_trace = uti.load_viewport(VIEWPORT_TRACE_FILENAME_NEW)

if __name__ == '__main__':
	main(FILE_LIST)