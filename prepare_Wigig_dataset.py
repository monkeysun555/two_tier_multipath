import numpy as np
import matplotlib.pyplot as plt


filename_list = ['./BW_Trace_5G_0.txt','./BW_Trace_5G_1.txt'\
					,'./BW_Trace_5G_2.txt','./BW_Trace_5G_3.txt'\
					,'./BW_Trace_5G_4.txt']

data_list = []
for i in filename_list:
	if i == './BW_Trace_5G_2.txt':
		multiple = 0.8
		addition = 80
	elif i == './BW_Trace_5G_1.txt':
		multiple = 1.7
		addition = -520
	else:
		multiple = 1
		addition = 0

	with open(i) as f:
		content = f.readlines()
	# content = [float(x.strip()) for x in content]
	content = [max(multiple * float(x.strip()) + addition, -(multiple * float(x.strip()) + addition)) for x in content]
	for l in range(len(content)):
		if content[l] >= 100:
			content[l] = int(content[l])
	data_list.append(content)

fig_num = 1
figs = []
for j in data_list:
	x_value = []
	for k in range(len(j)):
		x_value.append(float(k)/2.0)
	p = plt.figure(fig_num, figsize=(20,5))
	plt.plot(x_value, j, color='chocolate', label='Heterogeneous Network', linewidth=1.5,alpha=0.9)
	plt.grid(linestyle='dashed', axis='y',linewidth=1.5, color='gray')

	figs.append(p)
	# formated = [round(x,2) for x in j]
	np.savetxt('WiGig_' + str(fig_num) + '.txt', j, fmt='%1.2f')
	fig_num += 1

for fig in figs:
	fig.show()
raw_input()
