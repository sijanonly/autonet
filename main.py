import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from configparser import ConfigParser

import json
import PIL
import numpy as np

import torch
import torch.optim as optim
import torchvision
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


from feature_engine import prepare_train_test, sliding_window, MyDataset, transform_data
from config import PARAMConfig, NetworkConfig
from networks import ChildNetwork, PolicyNetwork
from train import TrainManager
from utils import set_seed, ActionSelection, DataLoader, reward_func, reward_func2, prepare_plot, prepare_figure, DictObject

N_EPISODE = 100
SEED = 0

# Ensuring reproducibility
set_seed(SEED)


def main(config):
    seq_length = 5
    total_rewards = []
    writer = SummaryWriter()

    is_entropy = config.entropy
    is_shuffle = config.shuffle
    ## TODO : remove config use dict only
    network_conf = NetworkConfig(
        input_size=int(config.input_size), hidden_size=int(config.hidden_size), num_steps=int(config.num_steps), action_space=int(config.action_space), learning_rate=float(config.learning_rate), beta=float(config.beta)
    )

    trainset, valset, testset = prepare_train_test()

    X_train, y_train = sliding_window(trainset, seq_length)
    X_val, y_val = sliding_window(valset, seq_length)
    X_test, y_test = sliding_window(testset, seq_length)
    train_loader = DataLoader(X_train, y_train)
    val_loader = DataLoader(X_val, y_val)
    test_loader = DataLoader(X_test, y_test)

    episode = 0
   
    policy_network = PolicyNetwork.from_dict(dict(network_conf._asdict()))
    print('current policy network', policy_network)

    while episode < N_EPISODE:
        initial_state = [[3, 8, 16]]
        logit_list = torch.empty(size=(0, network_conf.action_space))
        weighted_log_prob_list = torch.empty(size=(0,), dtype=torch.float)

        action, log_prob, logits = policy_network.get_action(initial_state)
      
        child_network = ChildNetwork.from_dict(action)
        criterion = torch.nn.MSELoss()
        optimizer = optim.SGD(child_network.parameters(), lr=0.001, momentum=0.9)

        train_manager = TrainManager(
            model=child_network, criterion=criterion, optimizer=optimizer
        )
        start_time = time.time()
        train_manager.train(train_loader, val_loader, is_shuffle)

        elapsed = time.time() - start_time
        signal = train_manager.avg_validation_loss
        print('signal is', signal)
        reward = reward_func2(signal)
        print('reward is', reward)      
        weighted_log_prob = log_prob * reward
        total_weighted_log_prob = torch.sum(weighted_log_prob).unsqueeze(dim=0)

        weighted_log_prob_list = torch.cat(
            (weighted_log_prob_list, total_weighted_log_prob), dim=0
        )
        logit_list = torch.cat((logit_list, logits), dim=0)
        # update the controller network
        policy_network.update(logit_list, weighted_log_prob_list, is_entropy)

        total_rewards.append(reward)
        
        #prepare metrics
        current_action = map(str, list(action.values()))
        action_str = "/".join(current_action)
        ActionSelection.update_selection(action_str)
        ActionSelection.update_reward(action_str, reward)
        print('current name mapping', ActionSelection.name_mapper)
        with open('logs/run.log', 'a') as run_file:
            run_file.write(json.dumps(ActionSelection.name_mapper))
            run_file.write('\n')
        #reporting
        current_action = f"Action selection (Hidden size:{action['n_hidden']}, #layers {action['n_layers']}, drop_prob {action['dropout_prob']})."
        current_run = "Runs_{}".format(episode + 1) + " " + current_action
        counter = 0
        for train_loss, val_loss in zip(train_manager.train_losses, train_manager.val_losses):
            writer.add_scalars(
                current_run, {"train_loss": train_loss, "val_loss": val_loss}, counter
            )
            counter += 1

        writer.add_scalar(
            tag="Average Return over {} episodes".format(N_EPISODE),
            scalar_value=np.mean(total_rewards),
            global_step=episode,
        )


        if is_entropy:
            writer.add_scalar(
                    tag=f"Entropy over time - BETA:{config.beta}", scalar_value=policy_network.entropy_mean, global_step=episode
            )
        writer.add_scalar(
            tag="Episode runtime", scalar_value=elapsed, global_step=episode
        )
 
        #
        # Prepare the plot
        plot_buf = prepare_figure(ActionSelection.action_selection, "Action Selection (n_hidden, n_layers, dropout)")

        #image = PIL.Image.open(plot_buf)

        #image = ToTensor()(image)  # .unsqueeze(0)

        #writer.add_image("Image 1", image, episode)
        writer.add_figure("Image 1", plot_buf, episode)
        # Prepare the plot
        plot_buf = prepare_figure(
            ActionSelection.reward_distribution(), "Reward Distribution per action selection (n_hidden, n_layers, dropout)"
        )

        #image = PIL.Image.open(plot_buf)

        #image = ToTensor()(image)  # .unsqueeze(0)

        #writer.add_image("Image 2", image, episode)
        writer.add_figure("Image 2", plot_buf, episode)

        print('\n\nEpisode {} completed \n\n'.format(episode+1))
        episode += 1

    writer.close()


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Training script for autonet",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--entropy', default=False, action='store_true', help='Enable exploration with entropy?(False by default)')
    parser.add_argument('--shuffle', default=False, action='store_true', help='Enable random shuffle on each epoch?(False by default)')
    parser.add_argument('--beta', default=0.1, type=float, help='Set beta for entropy')
    args = vars( parser.parse_args()) # get dictionary

    cfg_parser = ConfigParser()
    cfg_parser.read("config.ini")
    
    default_params = cfg_parser.items('default')

    config = dict(default_params)
    config.update({k: v for k, v in args.items() if v is not None})
    print('current config', config)
    #exit(0)
    config_obj = DictObject(config)


    main(config_obj)

