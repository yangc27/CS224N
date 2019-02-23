#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch

class Highway(torch.nn.Module):
    def __init__(self, dim):
        """
        In the constructor, we initiate two linear layers for W_proj and W_gate and assign them as member variables.
        @param dim: Input and output dimentionalities are the same for our case and they are both e_word according to
        the handout.
        """
        super(Highway, self).__init__()
        self.proj = torch.nn.Linear(dim, dim)
        self.gate = torch.nn.Linear(dim, dim)
        self.dim_out = dim
    
    def forward(self, X_conv):
        """
        In the forward function, we accpet the output of convolutional network X_conv and map to X_highway based on
        equation 10 in the handout.
        @param X_conv: output of CNN and input for the highway network, x_conv_out in the handout, tensor with shape of (batch_size, e_word).
        @returns X_highway: output of highway network based on equation 10, x_highway in the handout, tensor with shape of (batch_size, e_word).
        """
        X_proj = self.proj(X_conv).clamp(min = 0)
        # print("X_proj size: ", X_proj.size())
        X_gate = torch.sigmoid(self.gate(X_conv))
        # print("X_gate size: ", X_gate.size())
        X_highway = X_gate * X_proj + (1 - X_gate) * X_conv
        # print(self.proj.weight)
        # print(self.proj.bias)
        # print("X_proj is: ", X_proj)
        # print(self.gate.weight)
        # print(self.gate.bias)
        # print("X_gate is: ", X_gate)

        return X_highway


### END YOUR CODE 

