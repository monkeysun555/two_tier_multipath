import scipy.io as sio
import numpy as np

VIDEO_LEN = 300
NETWORK_TRACE_LEN = VIDEO_LEN 		# For 5G python
# NETWORK_TRACE_LEN = VIDEO_LEN + 100 		# For two-tier multipath and vp only
def load_5G_Data(fname, multiple, addition):
	with open(fname) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [multiple * float(x.strip()) + addition for x in content]
	# print(content, len(content))
	new_content = []
	for i in range(0,VIDEO_LEN):
		new_content.append((content[2*i]+content[2*i+1])/2)
	return content[:2*NETWORK_TRACE_LEN], new_content

def load_5G_latency(fname):
	with open(fname) as f:
		content = f.readlines()
	# you may also want to remove whitespace characters like `\n` at the end of each line
	content = [float(x.strip()) for x in content]
	# print(content, len(content))
	return content[:NETWORK_TRACE_LEN]