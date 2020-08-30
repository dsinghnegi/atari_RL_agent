from models import dqn
from models import dqn_dueling 
from models import a3c


def DQNAgent(dueling=False ,**kwargs):
	if dueling:
		print("Using Dueling DQN")
		return dqn_dueling.DQNAgent(**kwargs)
	print("Using DQN")
	return dqn.DQNAgent(**kwargs)

def A3C(**kwargs):
	agent=a3c.A3C(**kwargs)
	return agent

def EnvBatch(**kwargs):
	return a3c.EnvBatch(**kwargs)


