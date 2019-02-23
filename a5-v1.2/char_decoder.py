#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch
import torch.nn as nn

class CharDecoder(nn.Module):
    def __init__(self, hidden_size, char_embedding_size=50, target_vocab=None):
        """ Init Character Decoder.

        @param hidden_size (int): Hidden size of the decoder LSTM
        @param char_embedding_size (int): dimensionality of character embeddings
        @param target_vocab (VocabEntry): vocabulary for the target language. See vocab.py for documentation.
        """
        ### YOUR CODE HERE for part 2a
        ### TODO - Initialize as an nn.Module.
        ###      - Initialize the following variables:
        ###        self.charDecoder: LSTM. Please use nn.LSTM() to construct this.
        ###        self.char_output_projection: Linear layer, called W_{dec} and b_{dec} in the PDF
        ###        self.decoderCharEmb: Embedding matrix of character embeddings
        ###        self.target_vocab: vocabulary for the target language
        ###
        ### Hint: - Use target_vocab.char2id to access the character vocabulary for the target language.
        ###       - Set the padding_idx argument of the embedding matrix.
        ###       - Create a new Embedding layer. Do not reuse embeddings created in Part 1 of this assignment.
        super(CharDecoder, self).__init__()
        self.charDecoder = nn.LSTM(char_embedding_size, hidden_size)
        self.char_output_projection = nn.Linear(hidden_size, len(target_vocab.char2id))
        self.decoderCharEmb = nn.Embedding(len(target_vocab.char2id), char_embedding_size, padding_idx = target_vocab.char2id['<pad>'])
        self.target_vocab = target_vocab

        ### END YOUR CODE


    
    def forward(self, input, dec_hidden=None):
        """ Forward pass of character decoder.

        @param input: tensor of integers, shape (length, batch)
        @param dec_hidden: internal state of the LSTM before reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns scores: called s_t in the PDF, shape (length, batch, self.vocab_size)
        @returns dec_hidden: internal state of the LSTM after reading the input characters. A tuple of two tensors of shape (1, batch, hidden_size)
        """
        ### YOUR CODE HERE for part 2b
        ### TODO - Implement the forward pass of the character decoder.
        # print("(length, batch): ", input.size())
        x_char_embed = self.decoderCharEmb(input.permute(1, 0)) # (batch, seq_len/length, char_emb_size)
        # print("(batch, seq_len, char_emb_size): ", x_char_embed.size())
        # LSTM (charDecoder) first input (length, batch, char_emb_size), output is (length, batch, hidden_size) 
        # last_hidden and last_cell are (1, batch, hidden_size)
        output, (last_hidden, last_cell) = self.charDecoder(x_char_embed.permute(1, 0, 2), dec_hidden)
        # linear input (length, batch, in_features/hidden_size), output (batch, length, out_features/voc_len)
        scores = self.char_output_projection(output)
        dec_hidden = (last_hidden, last_cell)
        return scores, dec_hidden

        ### END YOUR CODE 


    def train_forward(self, char_sequence, dec_hidden=None):
        """ Forward computation during training.

        @param char_sequence: tensor of integers, shape (length, batch). Note that "length" here and in forward() need not be the same.
        @param dec_hidden: initial internal state of the LSTM, obtained from the output of the word-level decoder. A tuple of two tensors of shape (1, batch, hidden_size)

        @returns The cross-entropy loss, computed as the *sum* of cross-entropy losses of all the words in the batch.
        """
        ### YOUR CODE HERE for part 2c
        ### TODO - Implement training forward pass.
        ###
        ### Hint: - Make sure padding characters do not contribute to the cross-entropy loss.
        ###       - char_sequence corresponds to the sequence x_1 ... x_{n+1} from the handout (e.g., <START>,m,u,s,i,c,<END>).
        
        # input sequence is x_1 ... x_n, e.g. <START>,m,u,s,i,c 
        input_seq = char_sequence[:-1, :]
        scores, _ = self.forward(input_seq, dec_hidden) # score size is (batch, length, voc_len)
        (batch, length, voc_len) = scores.size()
        loss_input = scores.contiguous().view(batch * length, voc_len)

        # target for calculating loss, x_2 ... x_{n+1}, e.g. m,u,s,i,c,<END>
        target = char_sequence[1:, :]
        (batch, length) = target.size()
        loss_target = target.contiguous().view(batch * length)

        # get indice of pad token
        pad_token = self.target_vocab.char2id['<pad>']

        # calculate loss
        loss = nn.CrossEntropyLoss(ignore_index = pad_token, reduction = 'sum')
        loss_output = loss(loss_input, loss_target)

        return loss_output

        ### END YOUR CODE

    def decode_greedy(self, initialStates, device, max_length=21):
        """ Greedy decoding
        @param initialStates: initial internal state of the LSTM, a tuple of two tensors of size (1, batch, hidden_size)
        @param device: torch.device (indicates whether the model is on CPU or GPU)
        @param max_length: maximum length of words to decode

        @returns decodedWords: a list (of length batch) of strings, each of which has length <= max_length.
                              The decoded strings should NOT contain the start-of-word and end-of-word characters.
        """

        ### YOUR CODE HERE for part 2d
        ### TODO - Implement greedy decoding.
        ### Hints:
        ###      - Use target_vocab.char2id and target_vocab.id2char to convert between integers and characters
        ###      - Use torch.tensor(..., device=device) to turn a list of character indices into a tensor.
        ###      - We use curly brackets as start-of-word and end-of-word characters. That is, use the character '{' for <START> and '}' for <END>.
        ###        Their indices are self.target_vocab.start_of_word and self.target_vocab.end_of_word, respectively.
        
        (_, batch, _) = initialStates[0].size()
        # print("batch size: ", batch)
        output_word = []
        start_id = self.target_vocab.start_of_word
        # print("start id: ", start_id)
        current_char = torch.tensor([[start_id] * batch]) # (length, batch) where length = 1
        # print(current_char)
        m = nn.Softmax(dim = 2)
        dec_hidden = initialStates
        # print(self.target_vocab.id2char)

        for i in range(max_length):
            # print(current_char)
            scores, dec_hidden = self.forward(current_char, dec_hidden) # (length, batch, voc_size)
            # print("score size: ", scores.size())
            # print(scores)
            prob = m(scores) # (length, batch, voc_size)
            # print("softmax size: ", prob.size())
            # print(prob)
            pred_char_id = torch.argmax(prob, dim = 2) # (length, batch)
            # print("predicted char id size: ", pred_char_id.size())
            # print("predicted char id: ", pred_char_id)
            current_char_list = [self.target_vocab.id2char[char_id.item()] for char_id in pred_char_id.squeeze(0)]
            output_word.append(current_char_list)
            current_char = pred_char_id
        
        decoded_char_list = list(map(list, zip(*output_word)))

        decodedWords = []
        for char_list in decoded_char_list:
            word = ''
            for char in char_list:
                if char == '}':
                    break;
                word += char
            decodedWords.append(word)

        # print(decodedWords)  
        return decodedWords  
        


        ### END YOUR CODE