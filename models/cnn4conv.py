#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9

import torch
from torch import nn


def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels, track_running_stats=True),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class CNN4Conv(nn.Module):
    def __init__(self, in_channels, num_classes, args):
        super(CNN4Conv, self).__init__()
        in_channels = in_channels
        num_classes = num_classes
        hidden_size = 64
        
        if args.img_size == 32:
            self.emb_dim = hidden_size * 2 * 2
        elif args.img_size == 28:
            self.emb_dim = hidden_size
        else:
            raise NotImplemented
            
        self.features = nn.Sequential(
            conv3x3(in_channels, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size),
            conv3x3(hidden_size, hidden_size)
        )

        self.linear = nn.Linear(self.emb_dim, num_classes)
        self.linear.bias.data.fill_(0)

    def forward(self, x):
        features = self.features(x)
        features = features.view((features.size(0), -1))
        logits = self.linear(features)
        
        return logits, features
    
    def get_embedding_dim(self):
        return self.emb_dim