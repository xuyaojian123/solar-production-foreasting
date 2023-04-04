#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File  : models.py
@Author: XuYaoJian
@Date  : 2022/11/1 16:24
@Desc  : 
"""
from torch import nn
import torch
from torch.nn import Linear, GRU
from torch.nn.functional import relu

class GruModel(nn.Module):
    def __init__(self, settings):
        super(GruModel, self).__init__()
        self.output_len = settings["output_len"]
        self.hidC = settings["in_var"]
        self.hidR = 14
        self.out_dim = settings["out_var"]
        self.num_layers = settings["rnn_layer"]
        self.dropout = nn.Dropout(settings["dropout"])
        self.rnn = nn.GRU(input_size=self.hidC, hidden_size=self.hidR, num_layers=self.num_layers, batch_first=False)
        self.projection = Linear(self.hidR, self.out_dim)

    def forward(self, x_enc):
        '''
        :param x_enc: [batch, input_len , num_features]
        :return:
        '''
        x = torch.zeros([x_enc.shape[0], self.output_len, x_enc.shape[2]]).to(x_enc.device)
        x_enc = torch.cat((x_enc, x), 1)  # [batch, input_len + output_len, num_features]
        x_enc = x_enc.permute(1, 0, 2)  # [input_len + output_len, batch, num_features]
        rnn_out, _ = self.rnn(x_enc)
        dec = rnn_out.permute(1, 0, 2) # [batch, input_len + output_len, num_features]
        # sample = self.projection(dec)
        # sample = relu(self.projection(dec))
        sample = self.projection(self.dropout(dec))
        sample = sample[:, -self.output_len:, -self.out_dim:] # [B, L, 1]
        return sample.squeeze(2) #[batch, L]
