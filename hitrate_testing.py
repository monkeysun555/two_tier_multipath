import pickle
import numpy as np

HITRATE_DICT = './traces/hitrate/hitrate.p'

def main():
	hitrate = pickle.load(open(HITRATE_DICT, "rb"))
	print(hitrate[0])



if __name__ == '__main__':
	main()