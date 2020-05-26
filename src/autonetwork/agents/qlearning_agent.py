import random
import numpy as np

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
        state1Str = self.stateToString(state1)
        state2Str = self.stateToString(state2)
        
        old_q = self.Q.get((state1Str, action1), None)
        if old_q is None:
            self.Q[(state1Str, action1)] = reward 
        else:
            best_a= max([self.getQ(state2Str, a) for a in self.actions])
            temp=self.alpha*(reward+self.gamma*best_a-old_q)
            self.Q[(state1Str, action1)]+=temp
        return self.Q
        