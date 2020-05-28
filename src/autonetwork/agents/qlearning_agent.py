import random
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib import pylab
import matplotlib.gridspec as gridspec

import random

class Agent(object):  
        
    def __init__(self, actions):
        self.actions = actions
        self.num_actions = len(actions)

    def act(self, state):
        raise NotImplementedError
        
class QLearningAgent(Agent):
    
    def __init__(self, actions, epsilon=0.1, alpha=0.8, gamma=0.95):
        super(QLearningAgent, self).__init__(actions)
        
        ## TODO 1
        ## Initialize empty dictionary here
        ## In addition, initialize the value of epsilon, alpha and gamma
        self.Q={}
        self.epsilon=epsilon
        self.gamma=gamma
        self.alpha=alpha
        
        self._init_q_table()
        
        self.fig = pylab.figure(figsize=(10, 5))
        gs = gridspec.GridSpec(1, 1)
        self.ax = pylab.subplot(gs[:, 0])
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        
        self.ax.xaxis.set_visible(True)
        self.ax.yaxis.set_visible(True)
        self.ax.set_xticks(np.arange(-0.5, 9, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, 9, 1), minor=True)
        self.ax.grid(which="minor", color="w", linestyle="-", linewidth=1)
        self.ax.set_title("Q value variation over time")
        self.display_q_table()
        
        
    def _init_q_table(self):
#         self.q_table = {ind : np.zeros(9) for ind in np.arange(len(self.actions))}
        self.q_table = np.zeros((9,9))
    
    def display_q_table(self):
#         data_dict = {i : self.q_table[:,i] for i in np.arange(len(self.q_table))}
#         df = pd.DataFrame(data_dict)
#         print(df)
        img = np.array(self.q_table, copy=True)
        if not hasattr(self, 'imgplot'):
            self.imgplot = self.ax.imshow(img, cmap='Reds', interpolation='nearest',  vmin=0, vmax=100)
        else:
            self.imgplot.set_data(img)
    
        self.fig.canvas.draw()

    def stateToString(self, state):
        mystring = ""
        if np.isscalar(state):
            mystring = str(state)
        else:
            for digit in state:
                mystring += str(digit)
        return mystring  
     
    def getQ(self, state, action):
        return self.Q.get((state, action), 0.0)
    
    def act(self, state, epsilon=None):
        
        if epsilon:
            self.epsilon = epsilon
        
        stateStr = self.stateToString(state)
        exp_exp_tradeoff = random.uniform(0, 1)
#         print('trade is', exp_exp_tradeoff)
        ## If this number > greater than epsilon --> exploitation (taking the biggest Q value for this state)
#         dynamic exploration : increasing.
# reducing the epsilon with increasing time (more exploration at the beginning)
# random initial state:
        if exp_exp_tradeoff > self.epsilon:
#             q_values = [self.getQ(stateStr,a) for a in self.actions]
# #             print('q values', q_values)
#             action = np.argmax(q_values)
            q_values = {a: self.getQ(stateStr,a) for a in self.actions}
            max_q = max(q_values.values())
            # if we have multiple state with max q values, we choose randomly
            actions_with_max_q = [a for a, q in q_values.items() if q == max_q]
            action = np.random.choice(actions_with_max_q)

        # Else doing a random choice --> exploration
        else:
#             action = env.action_space.sample()
            actions = self.actions
            action = np.random.randint(0, len(actions))
        
#         print('action is ', action)
        return action
    
    def learn(self, state1, action1, reward, state2, done):
        """
        Q[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
        """
#         print('state 1 and 2', state1, state2, type(state1), state1.astype(int))
#         print('unrae', np.ravel_multi_index(state1.astype(int), (9,9)))
        state_ind = np.ravel_multi_index(state1.astype(int), (9,9))
        state_xy = state1.astype(int)
#         print('q table', self.q_table[state_xy[0]][state_xy[1]])
        state1Str = self.stateToString(state1)
        state2Str = self.stateToString(state2)

        old_q = self.Q.get((state1Str, action1), None)
        if old_q is None:
            self.Q[(state1Str, action1)] = reward 
        else:
            best_a= max([self.getQ(state2Str, a) for a in self.actions])
            temp=self.alpha*(reward+self.gamma*best_a-old_q)
            self.Q[(state1Str, action1)]+=temp
#         print('q is', self.Q)
#         self.q_table[action1][state_ind] = self.Q[(state1Str, action1)]
        self.q_table[state_xy[0]][state_xy[1]] = self.Q[(state1Str, action1)]
        self.display_q_table()
        
#         if done:
#             self._init_q_table()
        
        return self.Q
        