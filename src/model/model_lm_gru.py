"""
Borrow from homework3.
"""

import torch
import torch.nn as nn
import math

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
        self.rnn = nn.GRU(ninput, nhid, nlayers, dropout=drop_rate)
        ########################################
        self.decoder = nn.Linear(nhid, nvoc)
        self.nhid = nhid
        self.nlayers = nlayers
        self.init_weights()

    def init_weights(self):
        init_uniform = 1.0 / math.sqrt(self.nhid)
        self.encoder.weight.data.uniform_(-init_uniform, init_uniform)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_uniform, init_uniform)

    def forward(self, input, hidden):
        """
        :param input: (Sequence, Batch size, Input dimension)
        :return: output, hidden.
        """
        embeddings = self.drop(self.encoder(input)) # embedding = (seq_len, batch_size, ninput)

        # WRITE CODE HERE within two '#' bar
        ########################################
        # With embeddings, you can get your output here.
        # The ouput contains hidden state h_t from the last layer of the GRU, for each t.
        # Output = (seq_len, batch_size, nhid)
        # The hidden contains hidden state for t = seq_len, for each layer.
        # Hidden = (nlyaers, batch_size, nhid)
        output, hidden = self.rnn(embeddings, hidden)
        output = self.drop(output)  # output = (seq_len, batch_size, nhid)
        ########################################
        decoded = self.decoder(output) # decoded = (seq_len * batch_size, n_hidden)
        return decoded, hidden

    def init_hidden(self, batch_size):
        """
        Provide zero value initial hidden state.
        :return: Tensor w/ value 0, size (nlayers * batch_size * nhid)
        """
        tensor = next(self.parameters())
        return tensor.new_zeros(size=(self.nlayers, batch_size, self.nhid))

