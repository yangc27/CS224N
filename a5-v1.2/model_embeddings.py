#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
model_embeddings.py: Embeddings for the NMT model
Pencheng Yin <pcyin@cs.cmu.edu>
Sahil Chopra <schopra8@stanford.edu>
Anand Dhoot <anandd@stanford.edu>
Michael Hahn <mhahn2@stanford.edu>
"""

import torch.nn as nn

# Do not change these imports; your module names should be
#   `CNN` in the file `cnn.py`
#   `Highway` in the file `highway.py`
# Uncomment the following two imports once you're ready to run part 1(j)

from cnn import CNN
from highway import Highway

# End "do not change" 

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their CNN-based embeddings.
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layer for one language
        @param embed_size (int): Embedding size (dimensionality) for the output 
        @param vocab (VocabEntry): VocabEntry object. See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()

        ## A4 code
        # pad_token_idx = vocab.src['<pad>']
        # self.embeddings = nn.Embedding(len(vocab.src), embed_size, padding_idx=pad_token_idx)
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        char_embed_size = 50
        drop_rate = 0.3
        char_voc_len = len(vocab.char2id)
        
        self.embeddings = nn.Embedding(char_voc_len, char_embed_size)
        self.cnn = CNN(char_embed_size, embed_size)
        self.highway = Highway(embed_size)
        self.dropput = nn.Dropout(drop_rate)
        self.embed_size = embed_size
        self.vocab = vocab


        ### END YOUR CODE

    def forward(self, input):
        """
        Looks up character-based CNN embeddings for the words in a batch of sentences.
        @param input: Tensor of integers of shape (sentence_length, batch_size, max_word_length) where
            each integer is an index into the character vocabulary

        @param output: Tensor of shape (sentence_length, batch_size, embed_size), containing the 
            CNN-based embeddings for each word of the sentences in the batch
        """
        ## A4 code
        # output = self.embeddings(input)
        # return output
        ## End A4 code

        ### YOUR CODE HERE for part 1j
        # constants
        e_char = 50 # char embedding size

        # reshape input to size (sentence_length * batch_size, max_word_length)
        (sentence_length, batch_size, max_word_length) = input.size()
        # print("embedding input size: ", input.size())
        x = input.contiguous().view(-1, max_word_length)

        # look up char embedding
        # print("vocab length: ", len(self.vocab))
        # print(x)
        x_emd = self.embeddings(x) # size (sentence_length * batch_size, max_word_length, e_char)
        x_reshape = x_emd.permute(0, 2, 1)

        # pass cnn
        x_cnn_out = self.cnn.forward(x_reshape) # size (sentence_length * batch_size, e_word)

        # pass highway
        x_highway_out = self.highway.forward(x_cnn_out) # size stays the same

        # dropout
        x_out = self.dropput(x_highway_out)

        # reshape to get output
        e_word = self.highway.dim_out
        output = x_out.view(sentence_length, batch_size, e_word)

        return output

        ### END YOUR CODE

