import os
import io

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import torch


def set_seed(seed):
    # seed for hashing algorithm(e.g order of keys in Dict)
    os.environ['PYTHONHASHSEED']=str(seed)
    
    # fixing seed for numpy pseudo-random generator
    np.random.seed(seed)

    # fixing seed for pytorch random number generator 
    torch.manual_seed(seed)

    # enable deterministic CuDNN operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class ActionSelection:
    action_selection = defaultdict(int)
    reward_dist = defaultdict(list)
    name_mapper = defaultdict(str)

    def __init__(self):
        pass

    @classmethod
    def update_selection(cls, key):
        current_size = len(cls.name_mapper)
        if key not in cls.name_mapper:
            name = "N{}".format(current_size+1)
            cls.name_mapper[key] = name
        name = cls.name_mapper[key]
        cls.action_selection[name] += 1

    @classmethod
    def update_reward(cls, key, reward):
        name = cls.name_mapper[key]
        cls.reward_dist[name].append(reward)

    @classmethod
    def reward_distribution(cls):
        rewards = cls.reward_dist
        rewards_dict = defaultdict(float)
        for key, _ in rewards.items():
            rewards_dict[key] = sum(rewards[key]) / float(len(rewards[key]))
        
        return rewards_dict


class DataLoader:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    @property
    def input_size(self):
        return self.X.size(0)


def prepare_plot(action_dict, title):
    """Create a pyplot plot and save to buffer."""
    plt.figure(figsize=(9, 4))
    # plt.plot([1, 2])
    width = 0.1  # gives histogram aspect to the bar diagram
    print('action dict', action_dict)
    plt.bar(action_dict.keys(), action_dict.values(), width,align='edge', color=(0.2, 0.4, 0.6, 0.6))
    plt.xticks(rotation=10)
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format="jpeg")
    buf.seek(0)
    return buf


def prepare_figure(action_dict, title):
    """Create a pyplot plot and save to buffer."""
    fig = plt.figure(figsize=(9, 3))
    # plt.plot([1, 2])
    width = 0.1  # gives histogram aspect to the bar diagram
    #print('action dict', action_dict)
    plt.bar(action_dict.keys(), action_dict.values(), width, color=(0.2, 0.4, 0.6, 0.6))
    #plt.xticks(rotation=10)
    plt.title(title)
    #buf = io.BytesIO()
    #plt.savefig(buf, format="jpeg")
    #buf.seek(0)
    return fig


def reward_func(signal):
    """
    If signal is below 1(loss<1), the reward will be high.
    Otherwise, reward will be low.
    """
    if signal < 1:
        reward = -np.log(signal) + 5
    else:
        reward = np.exp(-(signal - 1))
    return reward

def reward_func2(signal):
    if np.isclose(0, signal):
        reward = 50
    elif signal < 1:
        reward = -np.log(signal) + round(1/(signal*10))
    else:
        reward = np.exp(-(signal - 1))

    return reward
