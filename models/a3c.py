import torch
import torch.nn as nn
from torch.nn import functional as F

from torch.autograd import Variable
import  numpy as np

def normalized_columns_initializer(weights, std=1.0):
	out = torch.randn(weights.size())
	out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
	return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class A3C(nn.Module):
	def __init__(self, n_actions, hidden=256, lstm=True):
		super().__init__()
		
		self.n_actions = n_actions
		self.hidden=hidden
		self.lstm=lstm
		self.input_dims=1 if self.lstm else 4
		self.network=nn.Sequential(
			nn.Conv2d(self.input_dims,32,3,2,1),
			nn.ReLU(),
			# nn.BatchNorm2d(32),

			nn.Conv2d(32,32,3,2,1),
			nn.ReLU(),
			# nn.BatchNorm2d(64),

			nn.Conv2d(32,32,3,2,1),
			nn.ReLU(),
			# nn.BatchNorm2d(64),

			nn.Conv2d(32,32,3,2,1),
			nn.ReLU(),
			
			nn.Flatten(),
		)

		if self.lstm:
			self.lstm_layer=nn.LSTMCell(288, self.hidden)
		else:
			self.linear_layer=nn.Linear(288,self.hidden)


		self.logits_network = nn.Sequential(
			nn.Linear(self.hidden,self.n_actions),
			)

		self.state_value_network = nn.Sequential(
			nn.Linear(self.hidden,1),
			)

		self.apply(weights_init)
		
		
		self.logits_network[-1].weight.data = normalized_columns_initializer(
			self.logits_network[-1].weight.data, 0.01)
		self.logits_network[-1].bias.data.fill_(0)
		
		self.state_value_network[-1].weight.data = normalized_columns_initializer(
			self.state_value_network[-1].weight.data, 1.0)
		
		self.state_value_network[-1].bias.data.fill_(0)

		if self.lstm:
			self.lstm_layer.bias_ih.data.fill_(0)
			self.lstm_layer.bias_hh.data.fill_(0)
		else:
			self.linear_layer.weight.data = normalized_columns_initializer(
					self.linear_layer.weight.data, 1.0)
		
			self.linear_layer.bias.data.fill_(0)

		self.train()
				 
	def forward(self, state_t, hidden_unit=None):
		model_device = next(self.parameters()).device
		state_t = torch.tensor(state_t, device=model_device, dtype=torch.float)
		
		x= self.network(state_t)
		
		if self.lstm:			
			if hidden_unit is None:
				cx = Variable(torch.zeros(state_t.size(0), 256).to(model_device))
				hx = Variable(torch.zeros(state_t.size(0), 256).to(model_device))
			else:
				(hx, cx)=hidden_unit
				hx=hx.detach()
				cx=cx.detach()

			x = x.view(x.size(0), -1)
			hx, cx=self.lstm_layer(x,(hx, cx))
			hidden_unit=(hx, cx)
			x=hx
		else:
			x=self.linear_layer(x)

		logits=self.logits_network(x)
		state_values=self.state_value_network(x)
		
		return (logits, state_values), hidden_unit


	def sample_actions(self, agent_outputs):
		logits, state_values = agent_outputs
		probs=F.softmax(logits,dim=1)
		action=probs.multinomial(num_samples=1).detach()
		return action

	def best_actions(self, agent_outputs):
		logits, state_values = agent_outputs
		probs=F.softmax(logits,dim=1)
		action=probs.max(1,keepdim=True)[1].detach().cpu().numpy()
		return action