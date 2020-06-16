from models import dqn
from models import dqn_dueling 

def DQNAgent(dueling=False ,**kwargs):
	if dueling:
		print("Using Dueling DQN")
		return dqn_dueling.DQNAgent(**kwargs)
	print("Using DQN")
	return dqn.DQNAgent(**kwargs)

