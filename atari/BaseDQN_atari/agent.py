from net import Net
from replayBuffer import ReplayBuffer
import numpy as np

# transfer learning agent

def linear_interp(min_step, max_step, it, final_p, init_p=1.):
    num = (max_step-min_step)*init_p + (init_p-final_p)*min_step - (init_p-final_p)*it
    denom = max_step-min_step
    v = num/denom
    return v

class Agent:
    
    def __init__(self, env, config):
        self.C = config
        self.n_state = list(env.observation_space.shape)
        self.n_action = env.action_space.n
        self.epsilon = 1.
        self.lr = 1e-3
        self.buffer = ReplayBuffer(self.C['max_size'],self.C['frame_stack'])
        self.net = Net(self.n_state, self.n_action, self.C)
    
        
    def act(self, s):
        a = self.greedy_act(s) if np.random.random() > self.epsilon else np.random.randint(self.n_action)
        return a
        
    def greedy_act(self, s):
        return self.net.action(s)
    
    def record(self, s, a, r, d, it):
        self.buffer.append(s, a, r, d)
        if it <= 1e6:
            
            self.epsilon = linear_interp(0, 1e6, it, 0.1, 1.0)
        else:
          
            self.epsilon = max(linear_interp(1e6, 32e6, it, 0.01, 0.1), 0.01)
         
        if it > self.C['training_start']:
            if it % self.C['train_freq'] == 0:
                self.lr = 1e-4
                self.net.train(self.buffer, self.lr)
                # print(Q)
                
            if it % self.C['update_target_freq'] == 0:
                self.net.update_target_network()
