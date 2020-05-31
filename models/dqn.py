import torch
import torch.nn as nn
import  numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)



class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape

        # Define your network body here. Please make sure agent is fully contained here
        # nn.Flatten() can be useful
        self.network=nn.Sequential(
            nn.Conv2d(4,32,3,2),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32,64,3,2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64,128,3,2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128,256,3,2),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(2304,1024),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Linear(1024,n_actions),

        )
        # self.network.apply(weights_init)

    def forward(self, state_t):
        # state_t[:,1,:,:]-=state_t[:,0,:,:]
        # state_t[:,2,:,:]-=state_t[:,0,:,:]
        # state_t[:,3,:,:]-=state_t[:,0,:,:]
        qvalues = self.network(state_t)
        return qvalues

    def get_qvalues(self, states):
        """
            like forward, but works on numpy arrays, not tensors
            """
        with torch.no_grad():
            model_device = next(self.parameters()).device
            states = torch.tensor(states, device=model_device, dtype=torch.float)
            qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)

