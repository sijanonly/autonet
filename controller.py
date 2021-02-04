import numpy as np
import torch
import torch.nn as nn


class Agent(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_steps=2):
        super(Agent, self).__init__()

        
        self.num_filter_option = 3
        self.filter_size_option = 3
        print('data inside', input_size, hidden_size)
        self.lstm1 = nn.LSTMCell(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, self.num_filter_option)
        self.num_steps = num_steps
        self.nhid = hidden_size
        self.hidden = self.init_hidden()

    def forward(self, input1):
        outputs = []
        h_t, c_t = self.hidden
        print('num setps', self.num_steps)
        for i in range(self.num_steps):
            h_t, c_t = self.lstm1(input1, (h_t, c_t))
            output = self.decoder(h_t)
            input1 = output
            print('output inside loop', output)
            outputs += [output]
        print('outputs is', outputs)
        outputs = torch.stack(outputs).squeeze(1)

        return outputs

    def init_hidden(self):
        h_t = torch.zeros(1, self.nhid, dtype=torch.float)
        c_t = torch.zeros(1, self.nhid, dtype=torch.float)

        return (h_t, c_t)
