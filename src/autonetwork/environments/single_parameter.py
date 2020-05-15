import numpy as np
import sys


class Environment(object):
    def reset(self):
        raise NotImplementedError("Inheriting classes must override reset.")

    def actions(self):
        raise NotImplementedError("Inheriting classes must override actions.")

    def step(self):
        raise NotImplementedError("Inheriting classes must override step")


class ActionSpace(Environment):
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

    def update_state(self, state, action):
        # UP = 0
        # RIGHT = 1
        # DOWN = 2
        # LEFT = 3

        x, y = np.unravel_index(state, self.shape)
        around_map = [state - 9, state + 1, state + 9, state - 1]
        selected = around_map[action]
        if selected < 0:
            return state

        if selected > (9 * 9 - 1):
            return state

        next_idx, next_idy = np.unravel_index(selected, self.shape)
        if action == 1 and next_idx > x or action == 3 and next_idx < x:
            return state

        return selected

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
        done = self.is_done(self.s)
        return (self._convert_state(self.s), reward, done, "")

    def render(self, mode="rgb_array", close=False):
        if close:
            return
        final = np.argmax(self.R)
        if mode == "rgb_array":
            maze = np.zeros((9, 9))
            maze[np.unravel_index(self.s, self.shape)] = 2.0
            maze[np.unravel_index(final, self.shape)] = 0.5
            img = np.array(maze, copy=True)
            return img
