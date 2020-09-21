import argparse

import torch
import torchvision
import torchvision.transforms as transforms

from feature_engine import prepare_train_test, sliding_window, MyDataset, transform_data
from policy_gradient import PolicyGradient
from config import PARAMConfig, NetworkConfig
from networks import ChildNetwork, PolicyNetwork
from train import TrainManager
from utils import ActionSelection, DataLoader


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
        INPUT_SIZE=3, HIDDEN_SIZE=64, NUM_STEPS=3, ACTION_SPACE=3, LEARNING_RATE=0.001
    )

    trainset, valset, testset = prepare_train_test()

    X_train, y_train = sliding_window(trainset, seq_length)
    X_val, y_val = sliding_window(valset, seq_length)
    X_test, y_test = sliding_window(testset, seq_length)
    train_loader = DataLoader(X_train, y_train)
    val_loader = DataLoader(X_val, y_val)
    test_loader = DataLoader(X_test, y_test)

    episode = 0
    total_rewards = deque([], maxlen=100)
    while episode < N_EPISODE:
        initial_state = [[3, 8, 16]]
        logit_list = torch.empty(size=(0, self.ACTION_SPACE), device=self.DEVICE)
        weighted_log_prob_list = torch.empty(
            size=(0,), dtype=torch.float, device=self.DEVICE
        )
        policy_network = PolicyNetwork.from_dict(dict(network_conf._asdict()))

        action, log_prob, logits = policy_network.get_action(initial_state)

        child_network = ChildNetwork.from_dict(action)
        criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

        train_manager = TrainManager(
            model=child_network, criterion=criterion, optimizer=optimizer
        )
        start_time = time.time()
        train_manager.train(train_loader, val_loader)

        elapsed = time.time() - start_time
        reward = train_manager.avg_validation_loss
        print("current rewards", rewards)
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
            tag="Entropy over time", scalar_value=entropy, global_step=episode
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
