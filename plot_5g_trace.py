# To plot 5G bandwidth traces
import matplotlib.pyplot as plt
import numpy as np
import utilities as uti

VIDEO_LEN = 300
filename_list = ['./traces/bandwidth/BW_Trace_5G_0.txt','./traces/bandwidth/BW_Trace_5G_1.txt'\
					,'./traces/bandwidth/BW_Trace_5G_2.txt','./traces/bandwidth/BW_Trace_5G_3.txt'\
					,'./traces/bandwidth/BW_Trace_5G_4.txt', './traces/bandwidth/BW_Trace_5G_5.txt']
###############       Disturbed              Unstable          Stable   ################
IS_SAVING = 1

def main():
	traces = []
	for i in filename_list:
		if i == './traces/bandwidth/BW_Trace_5G_5.txt':
			video_length = 450
			fname1 = './traces/bandwidth/BW_Trace_5G_0.txt'
			with open(fname1) as f:
				content1 = f.readlines()
			content1 = [float(x.strip()) for x in content1]
			content1 = content1[: 2*video_length/3]
			fname2 = './traces/bandwidth/BW_Trace_5G_2.txt'
			with open(fname2) as f:
				content2 = f.readlines()
			content2 = [0.8 * float(x.strip()) + 80 for x in content2]
			content2 = content2[: 2*video_length/3]
			content = np.concatenate((content2, content1), axis=0)

			fname3 = './traces/bandwidth/BW_Trace_5G_4.txt'
			with open(fname3) as f:
				content3 = f.readlines()
			content3 = [float(x.strip()) for x in content3]
			content3 = content3[:2*video_length/3]
			content = np.concatenate((content, content3), axis=0)
		else:
			if i == './traces/bandwidth/BW_Trace_5G_2.txt':
				multiple = 0.8
				addition = 80
			elif i == './traces/bandwidth/BW_Trace_5G_1.txt':
				multiple = 1.7
				addition = -520
			else:
				multiple = 1
				addition = 0
			with open(i) as f:
				content = f.readlines()
			# you may also want to remove whitespace characters like `\n` at the end of each line
			content = [multiple * float(x.strip()) + addition for x in content]
			content = content[:2*VIDEO_LEN]
		# print(content, len(content))
		traces.append(content)

	# print(traces)
	x_value = []
	for i in range(2,2*(VIDEO_LEN+1)):
		x_value.append(float(i)/2.0)

	new_x_value = []
	for i in range(2,2*(450+1)):
		new_x_value.append(float(i)/2.0)
	p = plt.figure(1, figsize=(20,5))
	# plt.tight_layout()
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), traces[1],'r-', label='Partially Disturbed', linewidth=2.0)
	plt.plot(x_value, traces[0], color='cornflowerblue', label='BW Trace 1', linewidth=2.5)
	# plt.plot(range(1,VIDEO_LEN+1), traces[i],'r-', label='Real-time Bandwidth', linewidth='1.2')

	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right',fontsize=30)
	# plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Throughput Trace')
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, 0, 1200])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(200, 1200+1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085,right=0.97)	
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')



	q = plt.figure(2, figsize=(20,5))
	# plt.tight_layout()
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), traces[1],'r-', label='Partially Disturbed', linewidth=2.0)
	plt.plot(x_value, traces[1], color='red', label='BW Trace 2', linewidth=2.5)
	# plt.plot(range(1,VIDEO_LEN+1), traces[i],'r-', label='Real-time Bandwidth', linewidth='1.2')

	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right',fontsize=30)
	# plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Throughput Trace')
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, 0, 1200])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(200, 1200+1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085, right=0.97)	
	plt.grid(linestyle='dashed', axis='y', linewidth=1.5,color='gray')


	r = plt.figure(3, figsize=(20,5))
	# plt.tight_layout()
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), traces[1],'r-', label='Partially Disturbed', linewidth=2.0)
	plt.plot(x_value, traces[2], color='chocolate', label='BW Trace 3', linewidth=2.5)
	# plt.plot(range(1,VIDEO_LEN+1), traces[i],'r-', label='Real-time Bandwidth', linewidth='1.2')

	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right',fontsize=30)
	# plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Throughput Trace')
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, 0, 1200])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(200, 1200+1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085, right=0.97)	
	plt.grid(linestyle='dashed', axis='y', linewidth=1.5,color='gray')


	s = plt.figure(4, figsize=(20,5))
	# plt.tight_layout()
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), traces[1],'r-', label='Partially Disturbed', linewidth=2.0)
	plt.plot(x_value, traces[3], color='plum', label='BW Trace 4', linewidth=2.5)
	# plt.plot(range(1,VIDEO_LEN+1), traces[i],'r-', label='Real-time Bandwidth', linewidth='1.2')

	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right',fontsize=30)
	# plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Throughput Trace')
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, 0, 1200])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(200, 1200+1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085,right=0.97)	
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')


	t = plt.figure(5, figsize=(20,5))
	# plt.tight_layout()
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), traces[1],'r-', label='Partially Disturbed', linewidth=2.0)
	plt.plot(x_value, traces[4], color='olive', label='BW Trace 5', linewidth=2.5)
	# plt.plot(range(1,VIDEO_LEN+1), traces[i],'r-', label='Real-time Bandwidth', linewidth='1.2')

	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right',fontsize=30)
	# plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Throughput Trace')
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, 0, 1200])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(200, 1200+1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.085,right=0.97)	
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')


	u = plt.figure(6, figsize=(20,5))
	# plt.tight_layout()
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), traces[1],'r-', label='Partially Disturbed', linewidth=2.0)
	plt.plot(x_value, traces[0], color='blue', label='BW Trace 1', linewidth=1.5,alpha=0.9)
	# plt.plot(x_value, traces[1], color='red', label='Disturbed', linewidth=1.5, alpha=0.9)
	# plt.plot(x_value, traces[2], color='red',label='BW Trace 3', linewidth=1.5, alpha=0.9)
	# plt.plot(x_value, traces[3], color='red', label='BW Trace 4', linewidth=1.5, alpha=0.9)
	plt.plot(x_value, traces[4], color='red', label='BW Trace 5', linewidth=1.5, alpha=0.7)

	# plt.plot(range(1,VIDEO_LEN+1), traces[i],'r-', label='Real-time Bandwidth', linewidth='1.2')

	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	plt.legend(loc='upper right',fontsize=24)
	# plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Throughput Trace')
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, VIDEO_LEN, 0, 1800])
	plt.xticks(np.arange(0, VIDEO_LEN+1, 50))
	plt.yticks(np.arange(200, 1000+1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.095,right=0.97)	
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')



	v = plt.figure(7, figsize=(20,5))
	# plt.tight_layout()
	# plt.plot(bw_result[1], bw_result[0],'b-',label='Predict Bandwidth')
	# plt.plot(range(1,VIDEO_LEN+1), traces[1],'r-', label='Partially Disturbed', linewidth=2.0)
	plt.plot(new_x_value, traces[5], color='chocolate', label='Heterogeneous Network', linewidth=1.5,alpha=0.9)
	# plt.plot(x_value, traces[1], color='red', label='Disturbed', linewidth=1.5, alpha=0.9)
	# plt.plot(x_value, traces[2], color='red',label='BW Trace 3', linewidth=1.5, alpha=0.9)
	# plt.plot(x_value, traces[3], color='red', label='BW Trace 4', linewidth=1.5, alpha=0.9)
	# plt.plot(x_value, traces[4], color='red', label='BW Trace 5', linewidth=1.5, alpha=0.7)

	# plt.plot(range(1,VIDEO_LEN+1), traces[i],'r-', label='Real-time Bandwidth', linewidth='1.2')

	### for plot gamma
	# plt.plot(range(1,VIDEO_LEN+1), network_trace,'r-',label='Average:690.7 Mbps  Peak: 885.07 Mbps  std:211.1')
	######
	# plt.plot(range(1,VIDEO_LEN+1), network_trace_aux[:VIDEO_LEN],'r-',label='5G Bandwidth', linewidth = 1)
	# plt.legend(loc='upper right',fontsize=24)
	# plt.title('Bandwidth Predict and Real Trace')
	# plt.title('5G Throughput Trace')
	plt.xlabel('Second', fontsize=30)
	plt.ylabel('Mbps', fontsize=30)
	plt.tick_params(axis='both', which='major', labelsize=30)
	plt.tick_params(axis='both', which='minor', labelsize=30)
	plt.axis([0, 450, 0, 900])
	plt.xticks(np.arange(0, 450+1, 50))
	plt.yticks(np.arange(200, 1000+1, 200))
	plt.gcf().subplots_adjust(bottom=0.20, left=0.095,right=0.97)	
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')

	# p.show()
	# q.show()
	# r.show()
	# s.show()
	# t.show()
	u.show()
	v.show()
	raw_input()

	if IS_SAVING:
		# p.savefig('./figures/5g/5G_1.eps', format='eps', dpi=1000, figsize=(30, 10))
		# q.savefig('./figures/5g/5g_2.eps', format='eps', dpi=1000, figsize=(30, 10))
		# q.savefig('./figures/5g/5g_3.eps', format='eps', dpi=1000, figsize=(30, 10))
		# r.savefig('./figures/5g/5g_4.eps', format='eps', dpi=1000, figsize=(30, 10))
		# s.savefig('./figures/5g/5g_5.eps', format='eps', dpi=1000, figsize=(30, 10))
		# u.savefig('./figures/5g/5g_agg.eps', format='eps', dpi=1000, figsize=(30, 10))
		v.savefig('./figures/5g/5g_misc.eps', format='eps', dpi=1000, figsize=(30, 10))


if __name__ == '__main__':
	main()