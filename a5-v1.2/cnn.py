#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch

class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        In the constructor, we initiate one convolutional layer and assign it as a member variable
        @param in_channels: number of channels for the input signal, e_char in the handout
        @param out_channels: number of channels produced by the convolution, f in the handout
        @param kernel_size: size of kernel, k in the handout
        """
        super(CNN, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 5)

    def forward(self, X):
        """
        In the forward function, we accept the padded char embedding (X_reshape in the handout),
        and generate the output of the convolutional network (x_conv_out in the handout).
        @param X: padded char embedding, X_reshape in the handout, size (batch_size, e_char, max_word_length)
        @returns X_conv_out: output of the convolutional network, size (batch_size, e_word)
        """
        X_conv = self.conv(X)
        X_conv_relu = X_conv.clamp(min = 0)
        # print("X_conv_relu: ", X_conv_relu.size())
        # print(X_conv_relu)
        X_conv_out = torch.max(X_conv_relu, dim = 2)[0]
        # print("X_conv_out: ", X_conv_out.size())
        # print(X_conv_out)
        return X_conv_out

### END YOUR CODE

