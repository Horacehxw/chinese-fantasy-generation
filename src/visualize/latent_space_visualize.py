"""
Some of the code snippets from open source project:

https://github.com/rohithreddy024/VAE-Text-Generation
"""

import numpy as np
import torch as torch
import os
from src.model.model_vae import VAE
import torch.nn.functional as F
import math
import argparse
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
import os.path as osp
import math
import random
from src.utils import word_dropout

import tensorflow as tf
import tensorboard as tb
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile

parser = argparse.ArgumentParser(description="Pytorch flat lstm VAE lm_model for unsuperviced text generation")
parser.add_argument("--teacher_forcing", type=float, default=1) # probability to do teacher forcing (default to every iter)
parser.add_argument("--kl_annealing", default="tanh",
                    help="kl_annealing method: 'linear', 'none', 'cyclic'， 'tanh' (default)， 'cyclic_tanh'")
parser.add_argument("--cycle", type=int, default=4,
                    help="number of cycles to divide the whole training duration. (cyclic kl annealing)")
parser.add_argument("--aggressive", action="store_true",
                    help="apply aggresive training on encoder (from paper lagging inference ...)")
parser.add_argument('--train_batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval batch size')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--n_hidden_G', type=int, default=512)
parser.add_argument('--n_layers_G', type=int, default=1)
parser.add_argument('--n_hidden_E', type=int, default=512)
parser.add_argument('--n_layers_E', type=int, default=1)
parser.add_argument('--rec_coef', type=float, default=1,
                    help="weight of reconstruction loss.")
parser.add_argument('--n_z', type=int, default=300)
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--gpu_device', type=int, default=0)
parser.add_argument('--n_highway_layers', type=int, default=2)
parser.add_argument('--n_embed', type=int, default=300)
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--seed', type=int, default=42,
                    help='set random seed')
parser.add_argument("--use_cws", action="store_true", help="use chinese word segementation")
parser.add_argument("--generate_only", action="store_true",
                    help="generate sample paragraphs from pretrained lm_model w/ the same setting")
parser.add_argument("--generation_decoder", default="greedy",
                    help="generation decoding algorithm: 'greedy' (default), 'sample', 'beam search'"
                    ) # generation decoding algorithm
parser.add_argument("--visualize_epoch", type=int, default=20,
                    help="the epoch to visualize")
parser.add_argument("--n_samples", type=int, default=100,
                    help="number of samples to generate latent variable")

opt = parser.parse_args()
print(opt)

from src.dataset import get_iterators
# Set the random seed manually for reproducibility.
torch.manual_seed(opt.seed)



gpu_id = opt.gpu_device   #str(gpu_device)
if torch.cuda.is_available():
    device = torch.device("cuda", gpu_id)
    print("Using GPU: {}".format(torch.cuda.get_device_name(gpu_id)))
else:
    device = torch.device("cpu")
    print("Using CPU")


##################################################
# set up tensorboard & log directory
experiment_list = ["{}:{}".format(k,v) for k,v in vars(opt).items() if k not in ["visualize_epoch", "gpu_device", "n_samples",
                                                                                 "generation_decoder", "generate_only"]]
experiment_path = osp.join(*experiment_list)

result_path = (Path(__file__)/"../../../results/flat_vae").resolve()
print("result path = {}".format(result_path))

log_dir = osp.join(result_path/"logs", experiment_path) # path to store tensorboard logging data for each experiment
tb_writer = SummaryWriter(log_dir)

# set up lm_model storage path
model_dir = osp.join(result_path/"models", experiment_path) # path to store best lm_model for each experiment
if not osp.exists(model_dir):
    os.makedirs(model_dir)

# set up generated txt path
txt_dir = osp.join(result_path/"txt", experiment_path) # path to store best lm_model for each experiment
if not osp.exists(txt_dir):
    os.makedirs(txt_dir)
##################################################

# Load dataset
train_iter, test_iter, fields = get_iterators(opt, device)
vocab = fields["text"].vocab
opt.n_vocab = len(vocab)
opt.fields = fields
train_batch_list = list(train_iter) # prepare for random data batch in aggressive training.
test_batch_list = list(test_iter)
pad_id = vocab.stoi[fields["text"].pad_token]
unk_id = vocab.stoi[fields["text"].unk_token]
init_id = vocab.stoi[fields["text"].init_token]
eos_id = vocab.stoi[fields["text"].eos_token]
special_tokens = [pad_id, init_id, eos_id]

# load lm_model
vae = VAE(opt, unk_id, special_tokens)
vae = vae.to(device)

##################################################

def visualize_tsne(batch_list, n_samples=100):
    print("Start computing embeddings.")
    latent_list = []
    author_list = []
    book_list = []
    with torch.no_grad():
        for _ in range(n_samples):
            batch = random.choice(batch_list)
            x, x_len = batch.text
            author_list.append(batch.author)
            book_list.append(batch.book)
            mu, logvar = vae.encoder(x,  x_len)
            z, kld = vae.reparameterize(mu, logvar)
            latent_list.append(z)
    latent_variables = torch.cat(latent_list, 0)
    books = list(torch.cat(book_list, 0).cpu().numpy())
    authors = list(torch.cat(author_list, 0).cpu().numpy())
    print("Start writing results.")
    tb_writer.add_embedding(latent_variables, metadata=books, global_step=opt.visualize_epoch, tag="book title embedding for epoch {}".format(opt.visualize_epoch))
    tb_writer.add_embedding(latent_variables, metadata=authors, global_step=opt.visualize_epoch, tag="author embedding for epoch {}".format(opt.visualize_epoch))


if __name__ == '__main__':
    model_path = osp.join(model_dir, "state_dict_{}.tar".format(opt.visualize_epoch))
    vae.load_state_dict(torch.load(model_path))
    visualize_tsne(test_batch_list, opt.n_samples)