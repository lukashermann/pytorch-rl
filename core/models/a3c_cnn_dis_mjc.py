from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from utils.init_weights import init_weights, normalized_columns_initializer
from core.model import Model

class A3CCnnDisMjcModel(Model):
    def __init__(self, args):
        super(A3CCnnDisMjcModel, self).__init__(args)
        # build model
        # 0. feature layers
        # Input Dim 64x64
        self.dof = args.dof
        self.action_dim = args.action_dim
        self.output_dims = self.action_dim * self.dof
        self.input_channels = args.input_channels

        self.conv1 = nn.Conv2d(self.input_channels, 16, kernel_size=8, stride=4) # NOTE: for pkg="atari"
        self.rl1   = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.rl2   = nn.ReLU()
        self.fc_3 = nn.Linear(6*6*32, self.hidden_dim)
        self.lstm  = nn.LSTMCell(self.hidden_dim, self.hidden_dim, 1)
        # 1. policy output
        self.policy_4 = nn.Linear(self.hidden_dim, self.output_dims)
        self.policy_5_list = []
        for i in range(self.dof):
            self.policy_5_list.append(nn.Softmax())
        # 2. value output
        self.value_4  = nn.Linear(self.hidden_dim, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)
        self.policy_4.weight.data = normalized_columns_initializer(self.policy_4.weight.data, 0.01)
        self.policy_4.bias.data.fill_(0)
        self.value_4.weight.data = normalized_columns_initializer(self.value_4.weight.data, 1.0)
        self.value_4.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        assert self.input_dims[1] == 64
        x = x.view(x.size(0), self.input_channels, self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = x.view(-1, 6*6*32)
        x = self.fc_3(x)
        x, c = self.lstm(x, lstm_hidden_vb)
        p = self.policy_4(x)
        p_list = []
        for i in range(self.dof):
            p_list.append(self.policy_5_list[i](p[:,i*self.action_dim : (i + 1) * self.action_dim]))
        v = self.value_4(x)
        return p_list, v, (x, c)
