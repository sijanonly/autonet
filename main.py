import time
import argparse

import numpy as np

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


from feature_engine import prepare_train_test, sliding_window, MyDataset, transform_data
from config import PARAMConfig, NetworkConfig
from networks import ChildNetwork, PolicyNetwork
from train import TrainManager
from utils import ActionSelection, DataLoader, reward_func

N_EPISODE = 2

class Params:
    NUM_EPOCHS = 5
    ALPHA = 5e-3  # learning rate
    BATCH_SIZE = 3  #
    HIDDEN_SIZE = 64  # number of hidden nodes we have in our child network
    BETA = 0.1  # the entropy bonus multiplier
    INPUT_SIZE = 3
    ACTION_SPACE = 3
    NUM_STEPS = 3  # for 3 params
    GAMMA = 0.99


def main():
    seq_length = 5
    total_rewards = []
    writer = SummaryWriter()

    network_conf = NetworkConfig(
        input_size=3, hidden_size=64, num_steps=3, action_space=3, learning_rate=0.001
    )

    trainset, valset, testset = prepare_train_test()

    X_train, y_train = sliding_window(trainset, seq_length)
    X_val, y_val = sliding_window(valset, seq_length)
    X_test, y_test = sliding_window(testset, seq_length)
    train_loader = DataLoader(X_train, y_train)
    val_loader = DataLoader(X_val, y_val)
    test_loader = DataLoader(X_test, y_test)

    episode = 0
   
    while episode < N_EPISODE:
        initial_state = [[3, 8, 16]]
        logit_list = torch.empty(size=(0, network_conf.action_space))
        weighted_log_prob_list = torch.empty(size=(0,), dtype=torch.float)
        policy_network = PolicyNetwork.from_dict(dict(network_conf._asdict()))

        action, log_prob, logits = policy_network.get_action(initial_state)
        print('action is', action)
        child_network = ChildNetwork.from_dict(action)
        criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(child_network.parameters(), lr=0.001, momentum=0.9)

        train_manager = TrainManager(
            model=child_network, criterion=criterion, optimizer=optimizer
        )
        start_time = time.time()
        train_manager.train(train_loader, val_loader)

        elapsed = time.time() - start_time
        signal = train_manager.avg_validation_loss
        reward = reward_func(signal)
        print("current rewards", reward)
        weighted_log_prob = log_prob * reward
        total_weighted_log_prob = torch.sum(weighted_log_prob).unsqueeze(dim=0)

        weighted_log_prob_list = torch.cat(
            (weighted_log_prob_list, total_weighted_log_prob), dim=0
        )
        logit_list = torch.cat((logit_list, logits), dim=0)
        # update the controller network
        policy_network.update(logit_list, weighted_log_prob_list)

        total_rewards.append(reward)

        writer.add_scalar(
            tag="Average Return over episodes",
            scalar_value=np.mean(total_rewards),
            global_step=episode,
        )

        writer.add_scalar(
            tag="Entropy over time", scalar_value=policy_network.entropy_mean, global_step=episode
        )
        writer.add_scalar(
            tag="Episode runtime", scalar_value=elapsed, global_step=episode
        )

        current_action = ActionSelection.action_selection
        for key, value in ActionSelection.action_selection.items():
            actions = key.split("/")
            label = "Action n_layers:{0}, n_hiddens :{1}".format(actions[1], actions[0])
            writer.add_histogram(
                "Action distribution- {}".format(label), value, episode
            )

        episode += 1

        writer.close()


if __name__ == "__main__":
    main()
