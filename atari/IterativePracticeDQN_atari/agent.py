from net import Net
from replayBuffer import ReplayBuffer
import numpy as np

#Linear interpolation function for updating epsilon
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
        self.epsilon = 0.99
        self.lr = 1e-3 #Learning rate
        self.buffer = ReplayBuffer(self.C['max_size'],self.C['frame_stack']) #Memory for RL-Training
        self.buffer2 = ReplayBuffer(self.C['max_size'],self.C['frame_stack']) #Memory for Practice
        self.net = Net(self.n_state, self.n_action, self.C)
        
    #Random action during Practice     
    def act_pre(self):
        a = np.random.randint(self.n_action)
        return a

    #Epsilon greedy action selection function      
    def act(self, s):
        a = self.greedy_act(s) if np.random.random() > self.epsilon else np.random.randint(self.n_action)
        return a
        
    def greedy_act(self, s):
        return self.net.action(s)

    #Practice without recording experiences  
    def practice(self):
        self.lr = 1e-3 
        self.net.pre_train(self.buffer, self.lr)
    
    #Records experiences and calls training functions
    def record(self, s, a, r, d, it, pre):
        
        #Variable pre is used to differentiate practice from RL training.
        if pre:
            self.buffer2.append(s, a, r, d)
            if it > self.C['pre_training_start']:
               if it % self.C['pre_train_freq'] == 0:
                self.lr = 1e-3
                self.net.pre_train(self.buffer, self.lr)
                  
        else:
            self.buffer.append(s, a, r, d) 
            if it <= 6e5:

                self.epsilon = linear_interp(0, 6e5, it, 0.1, 1.0)
            else:

                self.epsilon = max(linear_interp(6e5, 10e6, it, 0.01, 0.1), 0.01)

            if it > self.C['training_start']:
                if it % self.C['train_freq'] == 0:
                    self.lr = 1e-4 #Learning rate for RL training
                    self.net.train(self.buffer, self.lr)

                if it % self.C['update_target_freq'] == 0:
                    self.net.update_target_network()

