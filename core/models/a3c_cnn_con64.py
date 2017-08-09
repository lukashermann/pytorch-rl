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

class A3CCnnCon64Model(Model):
    def __init__(self, args):
        super(A3CCnnCon64Model, self).__init__(args)
        # build model
        # 0. feature layers
        # 64x64 input

        self.conv1 = nn.Conv2d(self.input_dims[0], 16, kernel_size=8, stride=4) # NOTE: for pkg="atari"
        self.rl1   = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.rl2   = nn.ReLU()
        self.fc_3 = nn.Linear(6*6*32, self.hidden_dim)

        # lstm
        if self.enable_lstm:
            self.lstm  = nn.LSTMCell(self.hidden_dim, self.hidden_dim, 1)
            self.lstm_v  = nn.LSTMCell(self.hidden_dim, self.hidden_dim, 1)

        # 1. policy output
        self.policy_5   = nn.Linear(self.hidden_dim, self.output_dims)
        self.policy_sig = nn.Linear(self.hidden_dim, self.output_dims)
        self.softplus = nn.Softplus()
        # 2. value output
        self.value_5    = nn.Linear(self.hidden_dim, 1)

        self._reset()

    def _init_weights(self):
        self.apply(init_weights)

        self.policy_5.weight.data = normalized_columns_initializer(self.policy_5.weight.data, 0.01)
        self.policy_5.bias.data.fill_(0)
        self.value_5.weight.data = normalized_columns_initializer(self.value_5.weight.data, 1.0)
        self.value_5.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.lstm_v.bias_ih.data.fill_(0)
        self.lstm_v.bias_hh.data.fill_(0)

    def forward(self, x, lstm_hidden_vb=None):
        """
        p = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        p = self.rl1(self.fc1(p))
        p = self.rl2(self.fc2(p))
        p = self.rl3(self.fc3(p))
        p = self.rl4(self.fc4(p))
        p = p.view(-1, self.hidden_dim)
        """
        assert self.input_dims[1] == 64
        x = x.view(x.size(0), self.input_dims[0], self.input_dims[1], self.input_dims[1])
        x = self.rl1(self.conv1(x))
        x = self.rl2(self.conv2(x))
        x = x.view(-1, 6*6*32)
        x = self.fc_3(x)

        if self.enable_lstm:
            p_, v_ = torch.split(lstm_hidden_vb[0],1)
            c_p, c_v = torch.split(lstm_hidden_vb[1],1)
            p, c_p = self.lstm(x, (p_, c_p))
        p_out = self.policy_5(p)
        sig = self.policy_sig(p)
        sig = self.softplus(sig)

        """
        v = x.view(x.size(0), self.input_dims[0] * self.input_dims[1])
        v = self.rl1_v(self.fc1_v(v))
        v = self.rl2_v(self.fc2_v(v))
        v = self.rl3_v(self.fc3_v(v))
        v = self.rl4_v(self.fc4_v(v))
        v = v.view(-1, self.hidden_dim)
        """

        if self.enable_lstm:
            v, c_v = self.lstm_v(x, (v_, c_v))
        v_out = self.value_5(v)

        if self.enable_lstm:
            return p_out, sig, v_out, (torch.cat((p,v),0), torch.cat((c_p, c_v),0))
        else:
            return p_out, sig, v_out
