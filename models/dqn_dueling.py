import torch
import torch.nn as nn
import  numpy as np

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data,gain=nn.init.calculate_gain('relu'))
    elif type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight,gain=nn.init.calculate_gain('relu'))
        m.bias.data.fill_(0.01)



class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0,hidden=512):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        self.hidden=hidden

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
            
            nn.Flatten(),
            nn.Linear(6272,self.hidden),
            # nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.value_network=nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden//2,1),
        )

        self.advantage_network=nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.hidden//2,n_actions),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # self.network.apply(weights_init)

    def forward(self, state_t):
        model_device = next(self.parameters()).device
        state_t = torch.tensor(state_t, device=model_device, dtype=torch.float)/128.0 -1.0 
        
        value_and_advantage = self.network(state_t)
        value,advantage= torch.split(value_and_advantage,self.hidden//2,1)
        value=self.value_network(value)
        advantage=self.advantage_network(advantage) 
        qvalues = value + (advantage- torch.mean(advantage, axis=1, keepdims=True))
        return qvalues

    def get_qvalues(self, states):
        with torch.no_grad():
            qvalues = self(states)
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

