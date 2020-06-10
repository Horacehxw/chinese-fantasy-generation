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
parser.add_argument('--gpu_device', type=int, default=0)
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

from src.dataset import get_iterators
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

result_path = (Path(__file__)/"../../../results/lm-LSTM").resolve()
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
# train_batch_list = list(train_iter) # prepare for random data batch in aggressive training.

# load lm_model
lm_model = LMModel(opt.n_vocab, opt.n_embed, opt.n_hidden, opt.n_layers)
lm_model = lm_model.to(device)
# load optimizer
optimizer = torch.optim.Adam(lm_model.parameters(), lr=opt.lr)

# loss
# sum up the loss for every entry, then divide it by number of non-padding elements.
padding_id = vocab.stoi[fields["text"].pad_token]
criterion = torch.nn.CrossEntropyLoss(reduction="sum",ignore_index=padding_id) # pad token will contribute 0 to loss

#################################################

def calc_batch_loss(batch, teacher_forcing=1):
    """
    Calculate the loss for given batch of data
    :param batch: batch of data.
    :param teacher_forcing: (float, default=1) rate of teacher forcing
    :return: loss tensor
    """
    x, x_len = batch.text
    data = x[:-1].clone()
    target = x[1:].clone()
    x_len -= 1 # ignore the "[EOS]" token
    hidden = None
    if random.random() < teacher_forcing:
        output, hidden = lm_model(data, x_len, hidden)
        loss = criterion(output.view(-1, opt.n_vocab), target.view(-1))
    else:
        loss = 0
        input_vector = data[:1] # "[SOS]" for each element
        for idx in range(target.size(0)):
            # the length here will not influence hidden state of rnn.
            logit, hidden = lm_model(input_vector, [1]*input_vector.size(1), hidden)
            topv, topi = logit.topk(1)
            input_vector = topi.squeeze(2).detach()
            loss += criterion(logit.view(-1, opt.n_vocab), target[idx])
    loss /= (target != padding_id).int().sum()
    return loss

# Evalutate Function
def evaluate():
    """
    Evaluate the current lm_model on validation set
    :return: Perplexity = exp(average cross-entropy loss)
    """
    lm_model.eval() # turn on evaluation mode
    valid_loss = []
    with torch.no_grad():
        for batch in test_iter:
            loss = calc_batch_loss(batch) # teacher forcing default to 1
            valid_loss.append(loss.item())
    return np.mean(valid_loss)

# Train Function
def train():
    lm_model.train() # turn on training mode
    train_loss = []
    for batch in train_iter:
        loss = calc_batch_loss(batch, opt.teacher_forcing)
        optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(lm_model.parameters(), opt.clip)
        optimizer.step()
        train_loss.append(loss.item())
    return np.mean(train_loss)

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

        model_path = osp.join(model_dir, "state_dict_{}.tar".format(epoch))
        torch.save(lm_model.state_dict(), model_path)
        if val_loss < best_loss:
            print("best_model !!!")
            best_loss = val_loss
            model_path = osp.join(model_dir, "state_dict_best.tar")
            torch.save(lm_model.state_dict(), model_path)

def generate_paragraph(lm_model, hidden, decode="greedy"):
    input_vector = fields["text"].numericalize([[fields["text"].init_token]], device=device)
    output_str = fields["text"].init_token
    with torch.no_grad():
        while input_vector.item() != vocab.stoi[fields["text"].eos_token]:
            logit, hidden = lm_model(input_vector, [1], hidden)
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
            # FIXME: sometimes the model failed to generate [EOS] token.
            if len(output_str) > 999:
                output_str = "[ERR]" + output_str
                break
    return output_str

def generate_paragraphs(lm_model, n_sampe=50, decode="greedy"):
    lm_model.to(device)
    lm_model.eval()
    for i in range(n_sampe):
        h_0 = torch.randn((lm_model.nlayers, 1, lm_model.nhid), device=device)
        c_0 = torch.randn((lm_model.nlayers, 1, lm_model.nhid), device=device)
        output_str = generate_paragraph(lm_model, (h_0, c_0), decode)
        tb_writer.add_text("Generate Paragraph", output_str, i)
        print(output_str)


if __name__ == '__main__':
    if not opt.generate_only:
        training()
    model_path = osp.join(model_dir, "state_dict_best.tar")
    lm_model.load_state_dict(torch.load(model_path))
    generate_paragraphs(lm_model, decode=opt.generation_decoder)