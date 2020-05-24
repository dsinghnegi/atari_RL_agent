import os
import argparse

import torch
import numpy as np
import gym

from models.dqn import DQNAgent
from preprocessing import BreakoutNoFrameskip as BNF
from utils.helper import evaluate





ENV_LIST=['BreakoutNoFrameskip-v4']

def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--environment", default="BreakoutNoFrameskip-v4" ,help="envirement to play")
	ap.add_argument("-c", "--checkpoint",required=True, default="checkpoint" ,help="checkpoint for agent")
	ap.add_argument("-v", "--video", default="videos" ,help="videos_dir")
	ap.add_argument("-n", "--n_lives", default=5 ,help="number of n_lives")

		
	opt = ap.parse_args()
	return opt



def main():
	opt=get_args()	

	assert opt.environment in ENV_LIST, \
		"Unsupported environment: {} \nSupported environemt: {}".format(opt.environment, ENV_LIST)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	env = BNF.make_env()
	state_shape = env.observation_space.shape
	n_actions = env.action_space.n
		

	agent = DQNAgent(state_shape, n_actions).to(device)
	agent.load_state_dict(torch.load(opt.checkpoint))
	agent.eval()

	env_monitor = gym.wrappers.Monitor(env, directory=opt.video, force=True)
	sessions = [evaluate(env_monitor, agent, n_games=opt.n_lives, greedy=True) for _ in range(10)]
	env_monitor.close()




if __name__ == '__main__':
	main()