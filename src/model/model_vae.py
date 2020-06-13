"""
Modified from open source project:

https://github.com/rohithreddy024/VAE-Text-Generation
"""

import torch as torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.utils import log_sum_exp, word_dropout
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import numpy as np

class Highway(nn.Module):
    def __init__(self, opt):
        super(Highway, self).__init__()
        self.n_layers = opt.n_highway_layers
        self.non_linear = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])
        self.linear = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])
        self.gate = nn.ModuleList([nn.Linear(opt.n_embed, opt.n_embed) for _ in range(self.n_layers)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in range(self.n_layers):
            gate = self.sigmoid(self.gate[layer](x))	        #Compute percentage of non linear information to be allowed for each element in x
            non_linear = F.relu(self.non_linear[layer](x))	#Compute non linear information
            linear = self.linear[layer](x)	                #Compute linear information
            x = gate*non_linear + (1-gate)*linear           #Combine non linear and linear information according to gate

        return x

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.highway = Highway(opt)
        self.n_hidden_E = opt.n_hidden_E
        self.n_layers_E = opt.n_layers_E
        self.n_embed = opt.n_embed
        self.n_z = opt.n_z
        self.embedding = nn.Embedding(opt.n_vocab, opt.n_embed)
        self.lstm = nn.LSTM(input_size=opt.n_embed, hidden_size=opt.n_hidden_E, num_layers=opt.n_layers_E, bidirectional=True)
        self.hidden_to_mu = nn.Linear(2 * opt.n_hidden_E, opt.n_z)
        self.hidden_to_logvar = nn.Linear(2 * opt.n_hidden_G, opt.n_z)
        self.hidden = None

    def forward(self, x, lengths):
        """
        :param x: (Sequence, Batch size, Input dimension)
        :param lengths: (list[int], length=batch_size), length of each sequence in minibatch
        :return:
            mu: mean of latent variable distribution
            logvar: log(variance) of latent variable distribution
        """
        x = self.embedding(x)
        n_seq, batch_size, n_embed = x.size()
        x = self.drop(self.highway(x))
        x = pack_padded_sequence(x, lengths)                         # Remove padded tokens.
        _, (self.hidden, _) = self.lstm(x)	                         # Exclude c_T and extract only h_T
        self.hidden = self.hidden.view(self.n_layers_E, 2, batch_size, self.n_hidden_E)
        self.hidden = self.hidden[-1]	                             # Select only the final layer of h_T
        # e_hidden = (batch_size, n_hidden_E * 2), merge hidden states into one.
        e_hidden = torch.cat(list(self.hidden), dim=1)	                 # Merge hidden states of both directions; check size
        mu = self.hidden_to_mu(e_hidden)  # Get mean of lantent z
        logvar = self.hidden_to_logvar(e_hidden)  # Get log variance of latent z
        return mu, logvar

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.n_hidden_G = opt.n_hidden_G
        self.n_layers_G = opt.n_layers_G
        self.n_z = opt.n_z
        self.n_embed = opt.n_embed
        self.embedding = nn.Embedding(opt.n_vocab, opt.n_embed)
        self.lstm = nn.LSTM(input_size=opt.n_embed+opt.n_z, hidden_size=opt.n_hidden_G, num_layers=opt.n_layers_G)
        self.fc = nn.Linear(opt.n_hidden_G, opt.n_vocab)
        self.hidden = None
        self.predict_hidden = opt.predict_hidden
        self.fc_hidden = nn.Linear(opt.n_z, opt.n_hidden_G*opt.n_layers_G, bias=False)

    def init_hidden(self, batch_size, z):
        if self.predict_hidden:
            c_0 = self.fc_hidden(z)
            c_0 = c_0.view(batch_size, self.n_layers_G, self.n_hidden_G) # extract layers
            c_0 = c_0.permute([1, 0, 2]) # (n_layers_G, batch_size, n_hidden_G)
            h_0 = torch.tanh(c_0)
            # h_0 = self.fc_hidden(z)
            # c_0 = h_0.new_zeros(h_0.size())
        else:
            tensor = next(self.parameters())
            h_0 = tensor.new_zeros(self.n_layers_G, batch_size, self.n_hidden_G)
            c_0 = tensor.new_zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        return (h_0, c_0)

    def forward(self, x, lengths, z, g_hidden = None):
        """
        :param x: input (seq_len, batch_size, n_embed)
        :param lengths: (list[int], length=batch_size), length of each sequence in minibatch
        :param z: (z_batch, nz) latent variable
        :param g_hidden:
        :return: logit, hidden(h_T,c_T)
        """
        x = self.embedding(x)
        n_seq, batch_size, n_embed = x.size()
        z_inp = torch.cat([z] * n_seq, 0).view(n_seq, batch_size, self.n_z)	    #Replicate z inorder to append same z at each time step
        x = torch.cat([x, z_inp], dim=2)	                                    #Append z to generator word input at each time step
        x = self.drop(x)
        if g_hidden is None:	                                    #if we are validating
            self.hidden = self.init_hidden(batch_size, z)
        else:					                                    #if we are training
            self.hidden = g_hidden

        #Get top layer of h_T at each time step and produce logit vector of vocabulary words
        x = pack_padded_sequence(x, lengths)
        output, self.hidden = self.lstm(x, self.hidden)
        output, _ = pad_packed_sequence(output)
        output = self.fc(output) # 不需要 resize, 该运算只和 last dimension 相关

        return output, self.hidden	                                #Also return complete (h_T, c_T) incase if we are generating

class VAE(nn.Module):
    def __init__(self, opt, unk_id, special_tokens):
        super(VAE, self).__init__()
        self.encoder = Encoder(opt)
        self.generator = Generator(opt)
        self.word_drop = opt.word_dropout
        self.n_z = opt.n_z
        self.unk_id, self.special_tokens = unk_id, special_tokens

    def forward(self, x, lengths):
        """
        Forward propagation in trianing stage
        :param x: input sentence to encode. (seq_len, batch_size)
        :param lengths: (list[int]) length of each element in x.
        :return:
            logit: output of generator LSTM at each time stamp.
            G_hidden: h_t & c_t hidden state of generator LSTM.
            kld: KL divergence between N(mu, exp(logvar)) and N(0,1).
        """
        G_inp = x[:-1].clone()
        # Only do word dropout for training stage.
        if self.training:
            G_inp = word_dropout(G_inp, self.unk_id, self.special_tokens, drop=self.word_drop)

        # check size of mu v.s. self.n_z
        mu, logvar = self.encoder(x, lengths)
        z, kld = self.reparameterize(mu, logvar)

        # G_inp is 1 shorter than x.
        logit, _ = self.generator(G_inp, lengths-1, z)
        return logit, kld

    @staticmethod
    def reparameterize(mu, logvar):
        z = torch.randn(mu.size(), device=mu.device)  # Noise sampled from ε ~ Normal(0,1)
        z = mu + z * torch.exp(0.5 * logvar)  # Reparameterization trick: Sample z = μ + ε*σ for backpropogation
        kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()  # Compute KL divergence loss
        return z, kld

    def mutual_info(self, x, lengths):
        """
        *modified from https://github.com/jxhe/vae-lagging-encoder*
        Calculate  the approximate mutual information between z & x under distribution q(z|x).
            I(x, z) =E_xE_{q(z|x)}log(q(z|x)) - E_xE_{q(z|x)}log(q(z))

        :param x: input sentence. (seq_len, batch_size)
        :param x: (list[int]) length of each sequence in batch.
        :return: (float) approximate mutual information. can be non-negative when n_z > 1.
        """
        mu, logvar = self.encoder(x, lengths)
        x_batch, nz = mu.size()
        neg_entropy = (-0.5 * nz * math.log(2 * math.pi) - 0.5 * (1 + logvar).sum(-1)).mean()

        # [z_batch, 1, nz]
        z, kld = self.reparameterize(mu, logvar)
        z = z.unsqueeze(1)

        # [1, x_batch, nz]
        mu, logvar = mu.unsqueeze(0), logvar.unsqueeze(0)
        var = logvar.exp()

        # (z_batch, x_batch, nz)
        dev = z - mu # dimension broadcast

        # (z_batch, x_batch)
        log_density = -0.5 * ((dev ** 2) / var).sum(dim=-1) - \
                      0.5 * (nz * math.log(2 * math.pi) + logvar.sum(-1))

        # log q(z): aggregate posterior
        # [z_batch]
        log_qz = log_sum_exp(log_density, dim=1) - math.log(x_batch)

        return (neg_entropy - log_qz.mean(-1)).item()
