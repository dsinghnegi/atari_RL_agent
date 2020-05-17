import os
from tqdm import tqdm
import argparse

from preprocessing import BreakoutNoFrameskip as BNF


ENV_LIST=['BreakoutNoFrameskip-v4']

def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--environment", default="BreakoutNoFrameskip-v4" ,help="envirement to play")
	opt = ap.parse_args()
	return opt



def main():
	opt=get_args()	

	assert opt.environment in ENV_LIST, \
		"Unsupported environment: {} \nSupported environemt: {}".format(opt.environment, ENV_LIST)


	env = BNF.make_env()






if __name__ == '__main__':
	main()