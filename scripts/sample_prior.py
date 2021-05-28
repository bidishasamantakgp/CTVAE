import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
from torch.autograd import Variable
import tqdm
from utils import *
from dataset import *

import argparse
import random
import time
from math import ceil
from tqdm import trange

from model import RNN_VAE
from flows import *

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c)'
)
parser.add_argument('--data', default='yelp',
                    help='name of dataset yelp, amazon. imdb')
parser.add_argument('--bert', default='../BERT',
                    help='name of bert embedding folder')
parser.add_argument('--gpu', default=True, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=True, action='store_true',
                    help='whether to save model or not')
parser.add_argument('--save_dir', default='models_yelp',
                    help='model_dir')
parser.add_argument('--with_bert', default=False, action='store_true',
                    help='if with bert sentence encodings')
parser.add_argument('--entire_z', default=False, action='store_true',
                    help='for feature dim')

parser.add_argument('--h_dim', default=200, type=int,
                    help='encoder convertion')
parser.add_argument('--h_dim_1', default=100, type=int,
                    help='hidden dimension of fetaure layer')
parser.add_argument('--z_dim', default=128, type=int,
                    help='whether to save model or not')
parser.add_argument('--z_dim_1', default=128, type=int,
                    help='whether to save model or not')
parser.add_argument('--f_emb_dim', default=2, type=int,
                    help='whether to save model or not')
parser.add_argument('--feature_dim', default=32, type=int,
                    help='whether to save model or not')
parser.add_argument('--p_word_dropout', default=0.3, type=float,
                    help='word dropout rate')
parser.add_argument('--beta1_D', default=0.5, type=float,
                    help='beta1 parameter of the Adam optimizer for the discriminator')
parser.add_argument('--beta2_D', default=0.9, type=float,
                    help='beta2 parameter of the Adam optimizer for the discriminator')

parser.add_argument("--flows", default=2, type=int)
parser.add_argument("--flow", default="MAF", type=str)
parser.add_argument("--mbsize", default=128, type=int,
                    help='encoder convertion')
args = parser.parse_args()
with_bert = args.with_bert


log_interval = 320
device = 'cuda' if args.gpu else 'cpu'
mb_size = 128

if args.data.lower() == 'imdb':
  dataset = Imdb_Dataset()
  MODEL_DIR = 'models_imdb/'

if args.data.lower() == 'yelp':
  dataset = Yelp_Dataset(mbsize=mb_size, repeat=False, shuffle=False)
  MODEL_DIR = 'models_yelp/'
  train_size = 443248
  val_size = 4000
  test_size = 1000

if args.data.lower() == 'amazon':
  dataset = Amazon_Dataset(mbsize=mb_size)
  MODEL_DIR = 'models_amazon/'

if with_bert:
  bert_sentence_embedding = torch.load("../BERT/"+args.data+"_sentence.pt")
  assert bert_sentence_embedding.size()[0] == test_size

flow = eval(args.flow)
flows = [flow(dim=args.z_dim) for _ in range(args.flows)]

model = RNN_VAE(args, dataset.n_vocab, flows,
                pretrained_embeddings=dataset.get_vocab_vectors(),
                freeze_embeddings=False,
                gpu=args.gpu,
                with_bert_encodings=with_bert
                )
model.load_state_dict(torch.load(args.save_dir+"/vae.bin"))
n_batches = ceil(train_size/mb_size)

model.eval()

z_star_list = list()
z_list = list()
labels_list = list()
inputs_list = list()
for batch in dataset.train_iter:
  inputs, labels, indices = batch.text, batch.label, batch.index
  if args.gpu:
    inputs, labels, indices = inputs.cuda(), labels.cuda(), indices.cuda()

  if with_bert:
    sentence_embeddings = torch.stack(
        [bert_sentence_embedding[idx] for idx in indices])
    if args.gpu:
      sentence_embeddings = sentence_embeddings.cuda()
  else:
    sentence_embeddings = None

  mu, logvar, z, z_star, z_star_prior_logprob, z_star_log_det, recon_loss, recon_loss_f_c, \
      recon_loss_f_s, kl_loss_z_prior,  z_decoded, z_log_det = \
      model.forward(inputs, labels, bert_inputs=sentence_embeddings)

  z_star_list.extend(z_star.detach().cpu().numpy())
  labels_list.extend(labels)

  # to get the sentence
  seq_len, mbsize = inputs.size()
  num_range = torch.arange(
      0, seq_len).type(torch.ByteTensor).repeat(mbsize, 1).transpose(0, 1)
  num_range = num_range.cuda() if model.gpu else num_range
  lengths = torch.mul((inputs.data == model.EOS_IDX), num_range).sum(dim=0)
  orig_sentences = [dataset.idxs2sentence(
      x) for x in inputs.transpose(0, 1).data.tolist()]
  for i in range(len(lengths)):
    inputs_list.append(orig_sentences[i][1:lengths[i]])


z_star_ = np.array(z_star_list)
labels_list = torch.stack(labels_list, axis=0).detach().cpu().numpy()
m = -1
max_senti = max(z_star_[:, m])
min_senti = min(z_star_[:, m])
print('max_senti:', max_senti)
print('min_senti:', min_senti)
exit()
num_samples = 1000
num_per_sample = 1

neg_sentences = []
pos_sentences = []


for i in trange(num_samples):
  z_star_c = Variable(torch.randn(1, args.z_dim_1))
  z_star_c_numpy = z_star_c.detach().cpu().numpy()

  z_star_c_numpy[0][m] = max_senti
  z_star_c = torch.FloatTensor(z_star_c_numpy)
  z_star_c = z_star_c.cuda() if args.gpu else z_star_c
  z_decoded, z_log_det = model.forward_decoder_z(z_star_c)

  for j in range(num_per_sample):
    # z_1 = model.sample_z(mu, logvar)
    #print("z_j:", z_1[0][:10])
    sample_idxs = model.sample_sentence(z_decoded, temp=0.8)
    sampled_sentence = dataset.idxs2sentence(sample_idxs)
    # print("prop:", p, "z:", z_star_c[0][m], sampled_sentence)
    neg_sentences.append(sampled_sentence)

  z_star_c_numpy[0][m] = min_senti
  z_star_c = torch.FloatTensor(z_star_c_numpy)
  z_star_c = z_star_c.cuda() if args.gpu else z_star_c
  z_decoded, z_log_det = model.forward_decoder_z(z_star_c)

  for j in range(num_per_sample):
    # z_1 = model.sample_z(mu, logvar)
    #print("z_j:", z_1[0][:10])
    sample_idxs = model.sample_sentence(z_decoded, temp=0.8)
    sampled_sentence = dataset.idxs2sentence(sample_idxs)
    # print("prop:", p, "z:", z_star_c[0][m], sampled_sentence)
    #sample_idxs = model.sample_sentence(mu, temp =0.1)
    pos_sentences.append(sampled_sentence)

with open(args.save_dir+'/neg_prior.txt', 'w') as f:
  for sent in neg_sentences:
    f.write(sent)
    f.write('\n')

with open(args.save_dir+'/pos_prior.txt', 'w') as f:
  for sent in pos_sentences:
    f.write(sent)
    f.write('\n')
