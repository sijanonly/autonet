
from collections import defaultdict

import numpy as np

class ActionSelection:
    action_selection = defaultdict(int)
    reward_dist = defaultdict(list)

    def __init__(self):
        pass

    @classmethod
    def update_selection(cls, key):
        cls.action_selection[key] += 1

    @classmethod
    def update_reward(cls, key, reward):
        print("cls.reward_dist", cls.reward_dist, key, reward, cls.reward_dist[key])
        cls.reward_dist[key].append(reward)

    @classmethod
    def reward_distribution(cls):
        rewards = cls.reward_dist
        rewards_dict = defaultdict(float)
        for key, _ in rewards.items():
            rewards_dict[key] = sum(rewards[key]) / float(len(rewards[key]))
        print("final disttt", rewards_dict)
        return rewards_dict


class DataLoader:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    @property
    def input_size(self):
        return self.X.size(0)


def generate_figure(action_dict, title):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    # plt.plot([1, 2])
    width = 1.0  # gives histogram aspect to the bar diagram
    plt.bar(action_dict.keys(), action_dict.values(), width, color=(0.2, 0.4, 0.6, 0.6))
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    return buf

def reward_func(signal):
    """
    If signal is below 1(loss<1), the reward will be high.
    Otherwise, reward will be low.
    """
    if signal < 1:
        reward = -np.log(signal) + 1
    else:
        reward = np.exp(-(signal - 1))
    return reward
