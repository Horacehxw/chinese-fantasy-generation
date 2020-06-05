import numpy as np
import torch as torch
import os
from src.model.model_lm import LMModel
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
parser.add_argument('--train_batch_size', type=int, default=128, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=64, metavar='N',
                    help='eval batch size')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--n_hidden', type=int, default=512)
parser.add_argument('--n_layers', type=int, default=2)
parser.add_argument('--word_dropout', type=float, default=0.5)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--gpu_device', type=int, default=1)
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

result_path = (Path(__file__)/"../../../results/lm_model").resolve()
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

# load lm_model
lm_model = LMModel(opt.n_vocab, opt.n_embed, opt.n_hidden, opt.n_layers)
lm_model = lm_model.to(device)
# load optimizer
optimizer = torch.optim.Adam(lm_model.parameters(), lr=opt.lr)

# loss
criterion = torch.nn.CrossEntropyLoss()

#################################################

def evaluate():
    """
    Evaluate the current lm_model on validation set
    :return: Perplexity = exp(average cross-entropy loss)
    """
    lm_model.eval() # turn on evaluation mode
    total_loss = 0
    num_items = 0
    with torch.no_grad():
        for batch in test_iter:
            x = batch.text
            hidden = lm_model.init_hidden(x.size()[1])
            data = x[:-1].clone()
            target = x[1:].clone()
            data = data.to(device)
            target = target.to(device)
            output, hidden = lm_model(data, hidden)
            loss = criterion(output.view(-1, opt.n_vocab), target.view(-1)) # loss for (seq_len*batch_size) outputs
            total_loss += loss.item() * len(data)
            num_items += len(data)
    avg_loss = total_loss / num_items
    return avg_loss

########################################
# WRITE CODE HERE within two '#' bar
########################################
# Train Function
def train():
    lm_model.train() # turn on training mode
    total_loss = 0
    num_items = 0


    for batch in train_iter:
        x = batch.text
        hidden = lm_model.init_hidden(x.size()[1])
        data = x[:-1].clone()
        target = x[1:].clone()
        data = data.to(device)
        target = target.to(device)
        hidden = hidden.detach() # avoid long gradient history tracking.
        optimizer.zero_grad()

        if random.random() < opt.teacher_forcing:
            output, hidden = lm_model(data, hidden)
            loss = criterion(output.view(-1, opt.n_vocab), target.view(-1))
        else:
            loss = 0
            input_vector = data[:1]
            for idx in range(target.size(0)):
                logit, hidden = lm_model(input_vector, hidden)
                topv, topi = logit.topk(1)
                input_vector = topi.squeeze(2).detach()
                loss += criterion(logit.view(-1, opt.n_vocab), target[idx])
            loss /= float(target.size(0))

        loss.backward()
        clip_grad_norm_(lm_model.parameters(), opt.clip)
        optimizer.step()

        total_loss += loss.item() * len(data)
        num_items += len(data)
    avg_loss = total_loss / num_items
    return avg_loss

def training():
    print("start training")

    ## Loop over epochs
    best_loss = math.inf
    for epoch in range(1, opt.epochs + 1):
        print("--------------------------------------------")
        print("epoch {}".format(epoch))
        train_loss = train()
        val_loss = evaluate()
        print("training perplexity = {:.3f}, training loss = {:.3f}".format(math.exp(train_loss), train_loss))
        print("validation perplexity = {:.3f}, validation loss = {:3f}".format(math.exp(val_loss), val_loss))

        tb_writer.add_scalar("Loss/train", train_loss, epoch)
        tb_writer.add_scalar("Perplexity/train", math.exp(train_loss), epoch)
        tb_writer.add_scalar("Loss/val", val_loss, epoch)
        tb_writer.add_scalar("Perplexity/val", math.exp(val_loss), epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = lm_model
            torch.save(best_model, osp.join(model_dir, "best_model.pt"))
            print("best_model !!!")

# def generate_paragraphs(vae, n_sample=50, decode="greedy"):
#     vae.to(device)
#     vae.eval()
#     generator = vae.generator
#     for i in range(n_sample):
#         z = torch.randn([1, opt.n_z], device=device)
#         G_inp = fields["text"].numericalize([[fields["text"].init_token]], device=device)
#         str = fields["text"].init_token
#         G_hidden = None # init with no hidden state
#         with torch.autograd.no_grad():
#             while G_inp[0][0].item() != vocab.stoi[fields["text"].eos_token]:
#                 G_inp = vae.embedding(G_inp)
#                 logit, G_hidden = generator(G_inp, z, G_hidden)
#                 probs = F.softmax(logit[0], dim=1)
#                 if decode == "greedy":
#                     topv, topi = logit.topk(1)
#                     G_inp = topi.squeeze().detach()
#                 elif decode == "sample":
#                     G_inp = torch.multinomial(probs, 1)
#                 elif decode == "beam search":
#                     raise NotImplementedError("Havn't Implement beam search decoding method.")
#                 else:
#                     raise AttributeError("Invalid decoding method")
#                 str += (vocab.itos[G_inp[0][0].item()])
#         str = str.encode('utf-8')
#         tb_writer.add_text("Generate Paragraph", str, i)

def generate_paragraph(lm_model, hidden, decode="greedy"):
    input_vector = fields["text"].numericalize([[fields["text"].init_token]], device=device)
    output_str = fields["text"].init_token
    with torch.no_grad():
        while input_vector.item() != vocab.stoi[fields["text"].eos_token]:
            logit, hidden = lm_model(input_vector, hidden)
            if decode == "greedy":
                topv, topi = logit.topk(1)
                input_vector = topi[0].detach()
            elif decode == "sample":
                probs = F.softmax(logit[0], dim=1)
                input_vector = torch.multinomial(probs, 1)
            elif decode == "beam search":
                raise NotImplementedError("Havn't Implement beam search decoding method.")
            else:
                raise AttributeError("Invalid decoding method")
            output_str += (vocab.itos[input_vector[0][0].item()])
    return output_str

def generate_paragraphs(lm_model, n_sampe=50, decode="greedy"):
    lm_model.to(device)
    lm_model.eval()
    for i in range(n_sampe):
        hidden = torch.randn(lm_model.init_hidden(1).size(), device=device)
        output_str = generate_paragraph(lm_model, hidden, decode)
        # output_str = output_str.encode('utf-8')
        tb_writer.add_text("Generate Paragraph", output_str, i)
        print(output_str)


if __name__ == '__main__':
    if not opt.generate_only:
        training()
    model_path = osp.join(model_dir, "best_model.pt")
    lm_model = torch.load(model_path)
    generate_paragraphs(lm_model, decode=opt.generation_decoder)