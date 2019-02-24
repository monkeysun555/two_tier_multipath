import pickle
import numpy as np
import matplotlib.pyplot as plt

HITRATE_DICT = './traces/hitrate/hitrate.p'

def main():
	# hitrate = pickle.load(open(HITRATE_DICT, "rb"))
	# print(hitrate[0])

	with open('./cmp_total_rewards.txt') as f:
		qoe = f.readlines()
	qoe = [float(x.strip()) for x in qoe]


	f = plt.figure()
	plt.hist(qoe, cumulative=True, label='CDF', range=(300, 450),
	         histtype='step', alpha=0.8, color='k')

	f.show()
	raw_input()


if __name__ == '__main__':
	main()