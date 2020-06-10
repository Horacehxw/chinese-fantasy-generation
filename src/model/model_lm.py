"""
Borrow from homework3.
"""

import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LMModel(nn.Module):
    # Language lm_model is composed of three parts: a word embedding layer, a rnn network and a output layer.
    # The word embedding layer have input as a sequence of word index (in the vocabulary) and output a sequence of vector where each one is a word embedding.
    # The rnn network has input of each word embedding and output a hidden feature corresponding to each word embedding.
    # The output layer has input as the hidden feature and output the probability of each word in the vocabulary.
    def __init__(self, nvoc, ninput, nhid, nlayers, drop_rate=0.5, attention=False, nhead=4):
        """
        :param nvoc: total number of words in vocabulary
        :param ninput: embedding dimension
        :param nhid: hidden representation dimension
        :param nlayers: layer of stacked RNNs
        """
        super(LMModel, self).__init__()
        self.drop = nn.Dropout(drop_rate)
        self.encoder = nn.Embedding(nvoc, ninput)
        # WRITE CODE HERE witnin two '#' bar
        ########################################
        # Add positional encoding
        ########################################
        # Construct you RNN lm_model here. You can add additional parameters to the function.
        # self.rnn = nn.GRU(ninput, nhid, nlayers, dropout=drop_rate)
        self.rnn = nn.LSTM(ninput, nhid, nlayers, dropout=drop_rate)
        ########################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.nhid = nhid
        self.nlayers = nlayers
        self.hidden = None

    def forward(self, input, lengths, hidden=None):
        """
        :param input: (Sequence, Batch size, Input dimension)
        :param lengths: (list[int], length=batch_size), length of each sequence in minibatch
        :param hidden: hidden state of rnn module
        :return:
            output: (seq_len, batch_size, n_voc) logit of predicted next token
            hidden: hidden state of rnn
        """
        seq_len, batch_size = input.size()

        embeddings = self.drop(self.encoder(input)) # embedding = (seq_len, batch_size, nembed)

        if hidden is None:
            self.init_hidden(batch_size)
        else:
            self.hidden = hidden

        # pack --> rnn --> pad.
        # Although not necessary, pack & padding can save memory and accelerate inference speed.
        # The model can skip processing the pad tokens.
        embeddings = pack_padded_sequence(embeddings, lengths)
        output, self.hidden = self.rnn(embeddings, self.hidden)
        output, _ = pad_packed_sequence(output)

        output = self.drop(output)  # output = (seq_len, batch_size, nhid)
        decoded = self.decoder(output) # decoded = (seq_len, batch_size, n_hidden)
        return decoded, self.hidden

    def init_hidden(self, batch_size):
        """
        Provide zero value initial hidden state.
        :return: Tensor w/ value 0, size (nlayers * batch_size * nhid)
        """
        tensor = next(self.parameters())
        h_0 = tensor.new_zeros(self.nlayers, batch_size, self.nhid)
        c_0 = tensor.new_zeros(self.nlayers, batch_size, self.nhid)
        self.hidden = h_0, c_0

