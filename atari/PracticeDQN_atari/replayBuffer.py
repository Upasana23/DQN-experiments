from collections import deque
import numpy as np

#Common replay buffer class for Practice and RL training
class ReplayBuffer:
    
    def __init__(self, max_size, frame_stack):
        self.b_state = deque([], maxlen=max_size)
        self.b_action = deque([], maxlen=max_size)
        self.b_reward = deque([], maxlen=max_size)
        self.b_done = deque([], maxlen=max_size)
        self.frame_stack = frame_stack
    
    def append(self, s, a, r, d):
        self.b_state.append(s)
        self.b_action.append(a)
        self.b_reward.append(r)
        self.b_done.append(d)
        
    #Generates random samples for RL training    
    def sample(self, size):
        indexes = np.random.choice(len(self.b_done) -  self.frame_stack - 1, size)
        sb, ab, rb, db, nsb = [], [], [], [], []
        for i in indexes:
            ab.append(self.b_action[i+3])
            rb.append(self.b_reward[i+3])
            db.append(self.b_done[i+3])
            state = [self.b_state[j] for j in range(i, i+self.frame_stack+1)]
            done = [self.b_done[j] for j in range(i, i+self.frame_stack+1)]
            s, ns = self._pad_state(np.asarray(state), np.asarray(done))
            sb.append(s)
            nsb.append(ns)
            
        return np.asarray(sb), np.asarray(ab), np.asarray(rb), np.asarray(db), np.asarray(nsb)

    #Generates random samples for Practice 	
    def samplePT(self, size):
        indexes = np.random.choice(len(self.b_done) -  self.frame_stack - 1, size)
        sb, nsb, ab = [], [], []
        for i in indexes:
            ab.append(self.b_action[i+3])
            state = [self.b_state[j] for j in range(i, i+self.frame_stack+1)]
            done = [self.b_done[j] for j in range(i, i+self.frame_stack+1)]
            s, ns = self._pad_state(np.asarray(state), np.asarray(done))
            sb.append(s)
            nsb.append(ns)
            
        return np.asarray(sb), np.asarray(ab), np.asarray(nsb)

    #Utility function to pad zeros to the frame stack        
    def _pad_state(self, state, done):
        for k in range(self.frame_stack - 2, -1, -1):
            if done[k]:
                state = np.copy(state)
                state[:k+1].fill(0)
                break
            
        s = state[:-1,:,:]
        ns = state[1:,:,:]
        return s.transpose((1,2,0)), ns.transpose((1,2,0))
            
            