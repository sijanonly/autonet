import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import one_hot, log_softmax, softmax
from torch.distributions import Categorical

from controller import Agent

class ChildNetwork(nn.Module):
    def __init__(
        self, n_hidden, n_layers, dropout_prob, input_size=1, output_size=1, seq_len=5
    ):
        super(ChildNetwork, self).__init__()
        # unpack the actions from the list
        # n_hidden, n_layers, dropout_prob = actions.tolist()

        self.n_hidden = int(n_hidden)

        self.seq_len = seq_len

        self.n_layers = int(n_layers)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers,
            dropout=dropout_prob,
        )

        self.linear = nn.Linear(in_features=self.n_hidden, out_features=output_size)

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
            torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        )

    def init_hidden(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden)
        c0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden)
        # return [t.cuda() for t in (h0, c0)]
        return h0, c0

    def forward(self, input):
        h0, c0 = self.hidden
        out, (hn, cn) = self.lstm(input, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


class PolicyNetwork:
    # hidden size, number of layers, dropout_probs
    ACTION_SPACE = torch.tensor(
        [[10, 50, 100, 150], [1, 2, 3, 4], [0.2, 0.4, 0.5, 0.7]]
    )

    def __init__(
        self, input_size, hidden_size, num_steps, action_space, learning_rate=0.001, beta=0.1
    ):
        self.agent = Agent(input_size, hidden_size, num_steps)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), learning_rate)
        self.beta = beta
        self.action_space = action_space

    def _predict(self, s):
        """
        Compute the action probabilities of state s using the learning model
        """
        return self.agent(torch.tensor(s).float())

    def _policy_loss(self, logits, current_action_probs, is_entropy=True):
        """
        Entropy will add exploration benefits.
        (more elaboration : https://github.com/dennybritz/reinforcement-learning/issues/34)

        At the beginning, all the actions will have same probability of occurrence.
        After some episodes of learning, few of the choices (actions) might have
        higher probability of selection, and entropy keeps on decreasing over time.
        Args:
            logits : For each child network, we have a new set of logits representing actions
            current_action_probs : log prob times return(rewards) for a given child network
        """
        self.loss = -1 * torch.mean(current_action_probs)
        if is_entropy:
            self.probs = softmax(logits, dim=1) + 1e-8
            self.entropy = -1 * torch.sum(self.probs * log_softmax(self.probs, dim=1), dim=1)
            self.entropy_mean = torch.mean(self.entropy, dim=0)
            self.entropy_bonus = -1 * self.beta * self.entropy_mean
            self.loss += self.entropy_bonus

    def update(self, logits, log_probs, is_entropy=True):
        """
        Update the weights of the policy network.

        """
        self._policy_loss(logits, log_probs, is_entropy)

        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_action(self, s):
        """
        Estimate the policy and sample an action, compute its log probability
        """
        # probs = self.predict(s)
        # action = torch.multinomial(probs, 1).item()
        # log_prob = torch.log(probs[action])
        # return action, log_prob
        # TODO : test with multinomial ???

        logits = self._predict(s)
        ind = Categorical(logits=logits).sample().unsqueeze(1)
        action_mask = one_hot(ind, num_classes=self.action_space)
        action_selection_prob = log_softmax(logits, dim=1)
        log_prob = torch.sum(action_mask.float() * action_selection_prob, dim=1)

        action = torch.gather(self.ACTION_SPACE, 1, ind).squeeze(1).numpy()
        print("current action is", action)
        action_dict = {
            "n_hidden": action[0],
            "n_layers": action[1],
            "dropout_prob": action[2],
        }
        return action_dict, log_prob, logits

    @classmethod
    def from_dict(cls, d):
        return cls(**d)
