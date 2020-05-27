import numpy as np
import sys


class Environment(object):
    def reset(self):
        raise NotImplementedError("Inheriting classes must override reset.")

    def actions(self):
        raise NotImplementedError("Inheriting classes must override actions.")

    def step(self):
        raise NotImplementedError("Inheriting classes must override step")

class ActionSpace(object):
    def __init__(self, actions):
        self.actions = actions
        self.n = len(actions)
    
    def sample(self):
        return np.random.randint(0, self.n)
    
class TwoParaEnv(Environment):
    
    def __init__(self, R):
        self.shape = (9, 9)
        # define state and action space
        self.S = range(81)
        self.action_space = ActionSpace(range(4))

        # define rewards
        self.R = R.ravel()

    def _convert_state(self, state):
        converted = np.unravel_index(state, self.shape)
        return np.asarray(list(converted), dtype=np.float32)
    
    def reset(self):
        self.s = 0
        self.is_reset = True
        return self._convert_state(self.s)
    
    def _check_valid_move(self, state, action, current_pos_x):
        next_idx, next_idy = np.unravel_index(state, self.shape)
        is_state_valid = state >0 or state < 9*9-1
        is_valid_move = (action == 1 and next_idx < current_pos_x) or (action==3 and next_idx > current_pos_x)
        
        return is_state_invalid and is_invalid_move
    
    def update_state(self, state, action):
        """
        Update state after action. If the moves are invalid, we will accumulate negative reward for each
        invalid move
        """
             # UP = 0
            # RIGHT = 1
            # DOWN = 2
            # LEFT = 3
#         print('inside update state: current state', state)
#         print('insdie update state : current action', action)
        invalid_move_reward = 0
        new_action = np.random.randint(0, self.action_space.n)
        x, y = np.unravel_index(state, self.shape)
        around_map = [state-9, state+1, state+9, state-1]
        selected = around_map[action]
        if selected < 0:
            invalid_move_reward -= 5
            return state
#             return self.update_state(state, new_action)
        
        if selected > (9*9-1):
            invalid_move_reward -= 5
#             return self.update_state(state, new_action)
            return state
        
        next_idx, next_idy = np.unravel_index(selected, self.shape)
        # if action is right and unravel idx is greather than current idx, it will to go next row,(stop there)
        if action == 1 and next_idx > x or action==3 and next_idx < x:
            invalid_move_reward -= 5
            return state
#             return self.update_state(state, new_action)
        
#         if self._check_valid_move(selected, action, x):
        return selected
#         else:
#             return self.update_state(state, new_action)
    
    def observe_reward(self, state, prev_state, action):
        if state == prev_state:
            return 0
        return self.R[state]
        
    def is_done(self, state):
        max_reward_idx = self.R.argmax()
        return state == max_reward_idx
        
    def step(self, action):
        s_prev = self.s
        self.s = self.update_state(self.s, action)
        reward = self.observe_reward(self.s, s_prev, action)
#         print('current step rewards', reward)
        done = self.is_done(self.s)
        return (self._convert_state(self.s), reward, done, '')
    
    def render(self, mode='rgb_array', close=False):
        if close:
            return
        final = np.argmax(self.R)
        max_value = np.max(self.R)
        R = self.R.copy()
        if mode == 'rgb_array':
            maze = np.zeros((9, 9))
#             print('current S is', self.s)
            maze = np.reshape(R, (-1, 9))
            maze[np.unravel_index(self.s, self.shape)] = max_value + 0.2
#             maze[np.unravel_index(final, self.shape)] = 200.0
            img = np.array(maze, copy=True)
            return img
