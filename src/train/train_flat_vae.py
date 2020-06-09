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

parser = argparse.ArgumentParser(description="Pytorch flat lstm VAE lm_model for unsuperviced text generation")
parser.add_argument("--teacher_forcing", type=float, default=1) # probability to do teacher forcing (default to every iter)
parser.add_argument("--kl_annealing", default="tanh",
                    help="kl_annealing method: 'linear', 'none', 'cyclic'， 'tanh' (default)， 'cyclic_tanh'")
parser.add_argument("--aggressive", action="store_true",
                    help="apply aggresive training on encoder (from paper lagging inference ...)")
parser.add_argument('--train_batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval batch size')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--n_hidden_G', type=int, default=512)
parser.add_argument('--n_layers_G', type=int, default=2)
parser.add_argument('--n_hidden_E', type=int, default=512)
parser.add_argument('--n_layers_E', type=int, default=1)
parser.add_argument('--rec_coef', type=float, default=1,
                    help="weight of reconstruction loss.")
parser.add_argument('--n_z', type=int, default=300)
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0003)
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

opt = parser.parse_args()
print(opt)

from src.dataset import get_itrerators
# Set the random seed manually for reproducibility.
torch.manual_seed(opt.seed)



gpu_id = opt.gpu_device   #str(gpu_device)
if torch.cuda.is_available():
    device = torch.device("cuda", gpu_id)
else:
    device = torch.device("cpu")


##################################################
# set up tensorboard & log directory
experiment_list = ["{}:{}".format(k,v) for k,v in vars(opt).items() if k not in ["gpu_device", "generation_decoder", "generate_only"]]
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
train_iter, test_iter, fields = get_itrerators(opt)
vocab = fields["text"].vocab
opt.n_vocab = len(vocab)
opt.fields = fields
train_batch_list = list(train_iter) # prepare for random data batch in aggressive training.

# load lm_model
vae = VAE(opt)
vae = vae.to(device)

# load optimizer
# TODO: learn internal algorithm of adam.
assert len(list(vae.encoder.parameters())) + len(list(vae.generator.parameters())) == \
       len(list(vae.parameters())), "The vae contains parameters not in encoder nor generator."
aggressive_flag = opt.aggressive
if aggressive_flag:
    enc_opt = torch.optim.Adam(vae.encoder.parameters(), lr=opt.lr)
    dec_opt = torch.optim.Adam(vae.generator.parameters(), lr=opt.lr)
else:
    vae_opt = torch.optim.Adam(vae.parameters(), lr=opt.lr)

# loss
# FIXME: ignore the padded index here!
cross_entropy = torch.nn.CrossEntropyLoss()

# KL weight function
num_iter = len(train_batch_list) * opt.epochs
duration = (num_iter // 4) + 1

def kl_weight(step):
    if opt.kl_annealing == "linear":
        kld_coef = min(1,step/float(duration))
    elif opt.kl_annealing == "cyclic":
        kld_coef = (step % duration) / duration
    elif opt.kl_annealing == "none" or opt.kl_annealing == "constant":
        kld_coef = 1
    elif opt.kl_annealing == "tanh":
        kld_coef = (math.tanh((step - duration//2) / 1000.0) + 1) / 2
    elif opt.kl_annealing == "cyclic_tanh":
        kld_coef = (math.tanh(((step%duration) - duration//2) / 1000.0) + 1) / 2
    else:
        raise NotImplementedError("Unkown kl annealing method.")
    return kld_coef

#################################################

# TODO: do we need to deal with [PAD] here?
def word_dropout(G_inp, drop=0.5):
    r = np.random.rand(G_inp.size(0), G_inp.size(1))
    # Perform word_dropout according to random values (r) generated for each word
    for i in range(len(G_inp)):
        for j in range(1, G_inp.size(1)):
            if r[i, j] < opt.word_dropout and G_inp[i, j] not in [vocab.stoi[fields["text"].eos_token],
                                                                  vocab.stoi[fields["text"].pad_token],
                                                                  vocab.stoi[fields["text"].init_token]
                                                                  ]:
                G_inp[i, j] = vocab.stoi[fields["text"].unk_token]
    return G_inp

# def train_batch_old(x, step, teacher_forcing=1):
#     # This is just like a latent variable + Language Model.
#     target = x[1:]  # target for generator should exclude first word of sequence
#     G_inp = x[:-1].clone()
#     G_inp = word_dropout(G_inp, opt.word_dropout)
#     if random.random() < teacher_forcing:
#         logit, _, kld = vae(x, G_inp)
#         rec_loss = cross_entropy(logit.view(-1, opt.n_vocab), target.contiguous().view(-1))
#     else:
#         rec_loss = 0
#         n_seq, batch_size = x.size()
#         mu, logvar = vae.encoder(x)
#         z, kld = vae.reparameterize(mu, logvar)
#         G_inp = G_inp[:1]
#         G_hidden = None
#         for di in range(target.size(0)):
#             logit, G_hidden = vae.generator(G_inp, z, G_hidden)
#             topv, topi = logit.topk(1)
#             G_inp = topi.squeeze(2).detach()
#             rec_loss += cross_entropy(logit.view(-1, opt.n_vocab), target[di])
#         rec_loss = rec_loss / float(target.size(0)) # average across each observation
#     kld_coef = kl_weight(step)
#     loss = opt.rec_coef * rec_loss + kld_coef*kld
#     vae_opt.zero_grad()
#     loss.backward()
#     clip_grad_norm_(vae.parameters(), opt.clip)         # gradient clipping
#     vae_opt.step()
#     # vae_scheduler.step()
#     return rec_loss.item(), kld.item()

def calc_batch_loss(x, kl_coef=1, teacher_forcing=1):
    """
    Calculate the loss of current model (train & eval status) on the gievn batch x and kl_weight.
    :param x: batch of data, each entry correspond to a word. (seq_len, batch_size)
    :param kl_coef: weight of KL divergence term in loss.
    :param teacher_forcing: rate of teacher forcing, default=1.
    :return:
        loss: (tensor) rec_loss + kl_weight * kl_loss
        rec_loss: (tensor) reconstruction loss. average across each ovservation.
        kl_loss: (tensor) KL divergence loss.
    """
    target = x[1:]  # target for generator should exclude first word of sequence
    G_inp = x[:-1].clone()
    G_inp = word_dropout(G_inp, opt.word_dropout)
    if random.random() < teacher_forcing: # teacher forcing
        logit, _, kld = vae(x, G_inp)
        rec_loss = cross_entropy(logit.view(-1, opt.n_vocab), target.contiguous().view(-1))
    else: # non teacher forcing
        rec_loss = 0
        mu, logvar = vae.encoder(x)
        z, kld = vae.reparameterize(mu, logvar)
        G_inp = G_inp[:1]
        G_hidden = None
        for di in range(target.size(0)):
            logit, G_hidden = vae.generator(G_inp, z, G_hidden)
            topv, topi = logit.topk(1)
            G_inp = topi.squeeze(2).detach()
            rec_loss += cross_entropy(logit.view(-1, opt.n_vocab), target[di])
        rec_loss = rec_loss / float(target.size(0))  # average across each observation
    loss = opt.rec_coef * rec_loss + kl_coef * kld
    return loss, rec_loss, kld

def train_aggressive(x, kl_coef=1, teacher_forcing=1, max_iter=100):
    """
    Train the encoder part aggressively
    :param x: *first* batch of data, each entry correspond to a word. (seq_len, batch_size)
    :param kl_coef: weight of KL divergence term in loss.
    :param teacher_forcing: rate of teacher forcing, default=1.
    :return: None
    """
    assert enc_opt is not None, "No encoder optimizer available."
    vae.train()
    burn_pre_loss = math.inf
    burn_cur_loss = 0 # running loss to validate convergence.
    for num_iter in range(1, max_iter+1):
        # update the encoder
        vae.zero_grad()
        loss, rec_loss, kl_loss = calc_batch_loss(x, kl_coef, teacher_forcing)
        burn_cur_loss += loss.item()
        loss.backward()
        clip_grad_norm_(vae.parameters(), opt.clip)
        enc_opt.step()

        # find next batch randomly
        batch = random.choice(train_batch_list)
        x = batch.text.to(device)

        # return in converge (15 borrowed from original opensource implementation)
        if num_iter % 15 == 0:
            if burn_cur_loss >= burn_pre_loss: # the smaller loss, the better.
                return
            burn_pre_loss = burn_cur_loss
            burn_cur_loss = 0

def evaluate():
    """
    Test the lm_model on validation dataset.
    :return: reconstruction loss, KL divergence loss
    """
    vae.eval()
    valid_rec_loss = []
    valid_kl_loss = []
    with torch.no_grad():
        for batch in test_iter:
            x = batch.text.to(device)
            loss, rec_loss, kld = calc_batch_loss(x) # kl_coef=1 by default.
            valid_rec_loss.append(rec_loss.item())
            valid_kl_loss.append(kld.item())
    return  np.mean(valid_rec_loss), np.mean(valid_kl_loss)

def calc_mi(data_iter):
    """
    Calculate the average mutual information between x & z in the given data iterator.
    :param data_iter: iterator of batch data.
    :return: (float) mutual_information.
    """
    mi = 0
    num_batches = 0
    with torch.no_grad():
        for batch in data_iter:
            x = batch.text.to(device)
            mi += vae.mutual_info(x)
            num_batches += 1
        mi /= float(num_batches)
    return mi

def training():
    global aggressive_flag
    print("start training")
    step = 0
    best_loss = math.inf
    pre_mutual_info = -math.inf

    for epoch in range(1, opt.epochs+1):
        train_rec_loss = []
        train_kl_loss = []
        train_mi = []
        for batch in train_iter:
            vae.train()
            runing_rec_loss = []
            runing_kl_loss = []
            runing_mi = []
            x = batch.text.to(device) 	                                #Used as encoder input as well as target output for generator
            kl_coef = kl_weight(step)

            # train encoder aggressively based on random mini-batches (current batch be the first one).
            if aggressive_flag:
                train_aggressive(x, kl_coef, teacher_forcing=opt.teacher_forcing)

            # calculate loss of model (after aggressive) training on current batch.
            loss, rec_loss, kl_loss = calc_batch_loss(x, kl_coef, opt.teacher_forcing)
            # zero out gradient for all parameters regardless of optimizer. (encoder+generator)
            vae.zero_grad()
            # back propagation & gradient clipping
            loss.backward()
            clip_grad_norm_(vae.parameters(), opt.clip)
            if aggressive_flag:
                dec_opt.step() # only update generator (encoder already trained aggressively).
            else:
                vae_opt.step() # train the whole model.

            # record losses in training
            mi = calc_mi([batch])
            rec_loss, kl_loss = rec_loss.item(), kl_loss.item()
            train_rec_loss.append(rec_loss)
            runing_rec_loss.append(rec_loss)
            train_kl_loss.append(kl_loss)
            runing_kl_loss.append(kl_loss)
            train_mi.append(mi)
            runing_mi.append(mi)
            step += 1
            if step % 100 == 0:
                runing_rec_loss = np.mean(runing_rec_loss)
                runing_kl_loss = np.mean(runing_kl_loss)
                runing_mi = np.mean(runing_mi)
                print("Iteration.", step, "T_ppl:", '%.2f' % np.exp(runing_rec_loss), "T_kld:", '%.2f' % runing_kl_loss,
                      "KL_weight:", kl_weight(step-1), "Mutual Info:", "%.2f" % runing_mi)
                tb_writer.add_scalar("Runing/KL_weight", kl_weight(step - 1), step)
                tb_writer.add_scalar("Runing/Perplexity", np.exp(runing_rec_loss), step)
                tb_writer.add_scalar("Runing/KL_loss", runing_kl_loss, step)
                tb_writer.add_scalar("Runing/Rec_loss", runing_rec_loss, step)
                tb_writer.add_scalar("Runing/ELBO_loss", runing_rec_loss+runing_kl_loss, step)
                tb_writer.add_scalar("Runing/Mutual_Info", runing_mi, step)

        train_rec_loss = np.mean(train_rec_loss)
        train_kl_loss = np.mean(train_kl_loss)
        train_mi = np.mean(train_mi)
        valid_rec_loss, valid_kl_loss = evaluate()
        vae.eval()
        valid_mi = calc_mi(test_iter)

        # stop aggressive training if the mutual information is not improved.
        if aggressive_flag:
            if pre_mutual_info >= valid_mi: # the greater mi, the better.
                aggressive_flag = False
                print("STOP BURNING")
            pre_mutual_info = valid_mi



        # may be report perplxity here.
        tb_writer.add_scalar("Rec_loss/train", train_rec_loss, epoch)
        tb_writer.add_scalar("Perplexity/train", np.exp(train_rec_loss), epoch)
        tb_writer.add_scalar("KL_loss/train", train_kl_loss, epoch)
        tb_writer.add_scalar("Loss/train", train_kl_loss+train_rec_loss, epoch)
        tb_writer.add_scalar("Rec_loss/val", valid_rec_loss, epoch)
        tb_writer.add_scalar("Perplexity/val", np.exp(valid_rec_loss), epoch)
        tb_writer.add_scalar("KL_loss/val", valid_kl_loss, epoch)
        tb_writer.add_scalar("Loss/val", valid_kl_loss+valid_rec_loss, epoch)
        tb_writer.add_scalar("Mutual_Information/train", train_mi, epoch)
        tb_writer.add_scalar("Mutual_Information/val", valid_mi, epoch)

        print("No.", epoch, "T_ppl:", '%.2f'%np.exp(train_rec_loss), "T_kld:", '%.2f'%train_kl_loss,
              "V_ppl:", '%.2f'%np.exp(valid_rec_loss), "V_kld:", '%.2f'%valid_kl_loss,
              "V_mi:", "%.2f"%valid_mi
              )
        model_path = osp.join(model_dir, "state_dict_{}.tar".format(epoch))
        torch.save(vae.state_dict(), model_path)
        if valid_kl_loss + valid_rec_loss < best_loss:
            print("Best Model !!!")
            best_loss = valid_kl_loss + valid_rec_loss
            model_path = osp.join(model_dir, "state_dict_best.tar")
            torch.save(vae.state_dict(), model_path)

def generate_paragraph(z, decode="greedy"):
    vae.eval()
    G_inp = fields["text"].numericalize([[fields["text"].init_token]], device=device)
    output_str = fields["text"].init_token
    G_hidden = None  # init with no hidden state
    with torch.no_grad():
        while G_inp[0][0].item() != vocab.stoi[fields["text"].eos_token]:
            logit, G_hidden = vae.generator(G_inp, z, G_hidden)
            if decode == "greedy":
                topv, topi = logit.topk(1)
                G_inp = topi[0].detach()
            elif decode == "sample":
                probs = F.softmax(logit[0], dim=1)
                G_inp = torch.multinomial(probs, 1)
            elif decode == "beam search":
                raise NotImplementedError("Havn't Implement beam search decoding method.")
            else:
                raise AttributeError("Invalid decoding method")
            output_str += (vocab.itos[G_inp[0][0].item()])
            # FIXME: sometimes the model failed to generate [EOS] token.
            if len(output_str) > 999:
                output_str = "[ERR]" + output_str
                break
    return output_str

def generate_paragraphs(vae, n_sample=50, decode="greedy"):
    vae.to(device)
    vae.eval()
    for i in range(n_sample):
        z = torch.randn([1, opt.n_z], device=device)
        output_str = generate_paragraph(z, decode)
        tb_writer.add_text("Generate Paragraph", output_str, i)
        print(output_str)

def visualize_graph():
    vae.to(device)
    vae.eval()
    batch = train_batch_list[0]
    x  = batch.text.to(device)
    G_inp = x[:-1].clone()
    tb_writer.add_graph(vae, (x, G_inp))


if __name__ == '__main__':
    if not opt.generate_only:
        visualize_graph()
        training()
    model_path = osp.join(model_dir, "state_dict_best.tar")
    vae.load_state_dict(torch.load(model_path))
    generate_paragraphs(vae, decode=opt.generation_decoder)