import scipy.io as sio
import numpy as np
import utilities as uti
import Wigig_streaming_dynamic_fix_cmp as wistr
import os

VIDEO_LEN = 45
COOKED_TRACE_FOLDER = './traces/e_f_hmm_sample/'
# COOKED_TRACE_FOLDER = './traces/single_test/'

def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
	cooked_files = os.listdir(cooked_trace_folder)
	all_cooked_bw = []
	for cooked_file in cooked_files:
		file_path = cooked_trace_folder + cooked_file
		cooked_bw = []
		with open(file_path, 'rb') as f:
			for line in f:
				parse = line.split()
				cooked_bw.append(float(parse[1]))
		new_cooked = [0.5*0.5*(cooked_bw[i] + cooked_bw[i+1]) for i in xrange(0,2*VIDEO_LEN+10,2)]
		# print(new_cooked)
		all_cooked_bw.append(new_cooked)

	return all_cooked_bw

def load_single_trace(cooked_trace_folder=COOKED_TRACE_FOLDER):
	cooked_files = os.listdir(cooked_trace_folder)
	all_cooked_bw = []
	count = 0
	for cooked_file in cooked_files:
		file_path = cooked_trace_folder + cooked_file
		cooked_bw = []
		with open(file_path, 'rb') as f:
			for line in f:
				parse = line.split()
				cooked_bw.append(float(parse[1]))
		all_cooked_bw.append(cooked_bw[:VIDEO_LEN+5])
		# count += 1
		# if count == 1:
		# 	break
	idx = np.random.randint(1, len(all_cooked_bw))

	return all_cooked_bw[idx]

def main():
	all_cooked_bw = load_trace()
	wistr.test_all(all_cooked_bw)

if __name__ == '__main__':
	main()