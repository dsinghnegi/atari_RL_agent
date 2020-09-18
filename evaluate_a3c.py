import os
import argparse

import torch
import numpy as np
import gym

from models import A3C
from utils.helper import evaluate_A3C, evaluate_A3C_lstm 

import environment 


def get_args():
	ap = argparse.ArgumentParser()
	ap.add_argument("-e", "--environment", default="BreakoutNoFrameskip-v4" ,help="envirement to play")
	ap.add_argument("-c", "--checkpoint",required=True ,help="checkpoint for agent")
	ap.add_argument("-v", "--video", default="videos" ,help="videos_dir")
	ap.add_argument("--lstm", action='store_true', help="Enable LSTM")
	ap.add_argument("--device", default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
		help="Device for training")

	opt = ap.parse_args()
	return opt


def main():
	opt=get_args()	

	assert opt.environment in environment.ENV_DICT.keys(), \
		"Unsupported environment: {} \nSupported environemts: {}".format(opt.environment, environment.ENV_DICT.keys())


	device = opt.device

	evaluate=evaluate_A3C_lstm 
	ENV=environment.ENV_DICT[opt.environment]

	env = ENV.make_env(clip_rewards=False)

	state_shape = env.observation_space.shape
	n_actions = env.action_space.n
	
	agent = A3C(n_actions=n_actions, lstm=opt.lstm).to(device)
	agent.load_state_dict(torch.load(opt.checkpoint, map_location=torch.device(device)))

	agent=agent.eval()

	env_monitor = gym.wrappers.Monitor(ENV.make_env(clip_rewards=False, lstm=opt.lstm), directory=opt.video, force=True)
	
	reward=evaluate(env_monitor, agent) 
	print("Reward: {}".format(reward))
	env_monitor.close()


if __name__ == '__main__':
	main()