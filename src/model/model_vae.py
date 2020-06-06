"""
Modified from open source project:

https://github.com/rohithreddy024/VAE-Text-Generation
"""

import torch as torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.lstm = nn.LSTM(input_size=opt.n_embed, hidden_size=opt.n_hidden_E, num_layers=opt.n_layers_E, bidirectional=True)

    def forward(self, x):
        n_seq, batch_size, n_embed = x.size()
        x = self.drop(self.highway(x))
        _, (self.hidden, _) = self.lstm(x)	                         #Exclude c_T and extract only h_T
        self.hidden = self.hidden.view(self.n_layers_E, 2, batch_size, self.n_hidden_E)
        self.hidden = self.hidden[-1]	                             #Select only the final layer of h_T
        # e_hidden = (batch_size, n_hidden_E * 2), merge hidden states into one.
        e_hidden = torch.cat(list(self.hidden), dim=1)	                 #merge hidden states of both directions; check size
        return e_hidden

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.drop = nn.Dropout(0.5)
        self.n_hidden_G = opt.n_hidden_G
        self.n_layers_G = opt.n_layers_G
        self.n_z = opt.n_z
        self.lstm = nn.LSTM(input_size=opt.n_embed+opt.n_z, hidden_size=opt.n_hidden_G, num_layers=opt.n_layers_G)
        self.fc = nn.Linear(opt.n_hidden_G, opt.n_vocab)

    def init_hidden(self, batch_size):
        tensor = next(self.parameters())
        h_0 = tensor.new_zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        c_0 = tensor.new_zeros(self.n_layers_G, batch_size, self.n_hidden_G)
        self.hidden = h_0, c_0

    def forward(self, x, z, g_hidden = None):
        """
        :param x: input (seq_len, batch_size, n_embed)
        :param z: latent variable
        :param g_hidden:
        :return: logit, hidden(h_T,c_T)
        """
        n_seq, batch_size, n_embed = x.size()
        z = torch.cat([z] * n_seq, 0).view(n_seq, batch_size, self.n_z)	    #Replicate z inorder to append same z at each time step
        x = torch.cat([x, z], dim=2)	                                    #Append z to generator word input at each time step
        x = self.drop(x)
        if g_hidden is None:	                                    #if we are validating
            self.init_hidden(batch_size)
        else:					                                    #if we are training
            self.hidden = g_hidden

        #Get top layer of h_T at each time step and produce logit vector of vocabulary words
        output, self.hidden = self.lstm(x, self.hidden)
        output = self.fc(output) # 不需要 resize, 该运算只和 last dimension 相关

        return output, self.hidden	                                #Also return complete (h_T, c_T) incase if we are generating


class VAE(nn.Module):
    def __init__(self, opt):
        super(VAE, self).__init__()
        self.embedding = nn.Embedding(opt.n_vocab, opt.n_embed)
        self.encoder = Encoder(opt)
        self.hidden_to_mu = nn.Linear(2*opt.n_hidden_E, opt.n_z)
        self.hidden_to_logvar = nn.Linear(2*opt.n_hidden_G, opt.n_z)
        self.generator = Generator(opt)
        self.n_z = opt.n_z

    def forward(self, x, G_inp, G_hidden = None):
        """
        Forward propagation in trianing stage
        :param x: input sentence to encode. (seq_len, batch_size)
        :param G_inp: input sentence for teacher forcing training (x[:-1]). (seq_len-1, batch_size)
        :param G_hidden: hidden state for generator, default=None.
        :return:
            logit: output of generator LSTM at each time stamp.
            G_hidden: h_t & c_t hidden state of generator LSTM.
            kld: KL divergence between N(mu, exp(logvar)) and N(0,1).
        """
        n_seq, batch_size = x.size()
        mu, logvar = self.encode(x)
        device = (next(self.parameters())).device
        z = torch.randn([batch_size, self.n_z], device=device)	            #Noise sampled from ε ~ Normal(0,1)
        z = mu + z * torch.exp(0.5 * logvar)	                            #Reparameterization trick: Sample z = μ + ε*σ for backpropogation
        kld = -0.5 * torch.sum(logvar - mu.pow(2) - logvar.exp() + 1, 1).mean()	#Compute KL divergence loss
        logit, G_hidden = self.generate(G_inp, z, G_hidden)
        return logit, G_hidden, kld

    def encode(self, x):
        """
        Given input sentence x, provide the latent variable's mu & log variance
        :param x: input sequence. (seq_len, batch_size)
        :return: mu, logvar
        """
        x = self.embedding(x)  # Produce embeddings from encoder input
        E_hidden = self.encoder(x)  # Get h_T of Encoder (batch, n_hidden_E * 2)
        mu = self.hidden_to_mu(E_hidden)  # Get mean of lantent z
        logvar = self.hidden_to_logvar(E_hidden)  # Get log variance of latent z
        return mu, logvar

    def generate(self, G_inp, z, G_hidden):
        """
        Given input sequence G_inp, latent vaiable z, perform VAE generator function
        :param G_inp: input sentence G_inp (should be x[:-1] in teacher forcing inference)
        :param z: latent variable sampled from encode latent space
        :param G_hidden:
        :return:
            logit: output of generator LSTM at each time stamp.
            G_hidden: h_t & c_t hidden state of generator LSTM.
        """
        G_inp = self.embedding(G_inp)  # Produce embeddings for generator input
        logit, G_hidden = self.generator(G_inp, z, G_hidden)
        return logit, G_hidden