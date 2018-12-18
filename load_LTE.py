import scipy.io as sio
import numpy as np
import utilities_lte as uti
import matplotlib.pyplot as plt


FILE_LIST = ['./traces/bandwidth/bus_test_1.txt']
VIDEO_LENGTH = 450

IS_SAVING = 1

def load_lte_data(fname, video_length, addition_length = 0):
	with open(fname) as f:
		content = f.readlines()
	# print(content)
	# you may also want to remove whitespace characters like `\n` at the end of each line
	# content = content[0].split('\r')
	content = [float(x.strip('\n')) for x in content]
	return content[:video_length]

def main(fnames):
	for fname in fnames:
		print(fname)
		network_trace = load_lte_data(fname, VIDEO_LENGTH)
		average_bw = uti.show_network(network_trace)
		print("<====================>")
	# yaw_trace, pitch_trace = uti.load_viewport(VIEWPORT_TRACE_FILENAME_NEW)

	p = plt.figure(1, figsize=(20,5))
	plt.plot( range(len(network_trace)), network_trace, color='cornflowerblue', label='LTE Bandwidth', linewidth=2.5)	
	plt.legend(loc='upper right',fontsize=30)
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, len(network_trace), 0, max(network_trace)*1.2])
	plt.xticks(np.arange(0, len(network_trace)+1, 50))
	plt.yticks(np.arange(0, max(network_trace)*1.2, 5))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085,right=0.97)	
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')
	p.show()
	raw_input()
	if IS_SAVING:
		p.savefig('./figures/lte/lte_trace/test_1.eps', format='eps', dpi=1000, figsize=(30, 10))


if __name__ == '__main__':
	main(FILE_LIST)