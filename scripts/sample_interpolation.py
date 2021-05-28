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

from model_flow_imdb_1 import RNN_VAE
from flows import *

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c)'
)
parser.add_argument('--data', default='imdb',
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
parser.add_argument('--output_dir')
parser.add_argument('--h_dim', default=200, type=int,
                    help='encoder convertion')
parser.add_argument('--h_dim_1', default=100, type=int,
                    help='hidden dimension of fetaure layer')
parser.add_argument('--z_dim', default=128, type=int,
                    help='whether to save model or not')
parser.add_argument('--z_dim_1', default=128, type=int,
                    help='whether to save model or not')
parser.add_argument('--f_emb_dim', default=3, type=int,
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
#parser.add_argument("--flow", default="MAF", type=str)
parser.add_argument("--flow", default="RealNVP", type=str)
parser.add_argument("--mbsize", default=128, type=int,
                    help='encoder convertion')
args = parser.parse_args()
with_bert = args.with_bert


lr = 0.001
lr_D = 1e-3
lr_decay_every = 1000000
n_iter = 138520
log_interval = 320
device = 'cuda' if args.gpu else 'cpu'
mb_size = 128

if args.data.lower() == 'imdb':
  dataset = Imdb_Dataset(mbsize=mb_size, repeat=False, shuffle=False)
  MODEL_DIR = 'models_imdb/'
  train_size = 708929
  val_size = 4000
  test_size = 1000

if args.data.lower() == 'yelp':
  dataset = Yelp_Dataset(mbsize=mb_size, repeat=False, shuffle=False)
  MODEL_DIR = 'models_yelp/'
  train_size = 443248
  val_size = 4000

if args.data.lower() == 'amazon':
  dataset = Amazon_Dataset(mbsize=mb_size, repeat=False, shuffle=False)
  MODEL_DIR = 'models_amazon/'

if with_bert:

  bert_sentence_embedding = torch.load(
      "../BERT/"+args.data+"_test_sentence.pt")
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
# '''
for batch in dataset.test_iter:
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

  z, z_star, z_star_prior_logprob, z_star_log_det, recon_loss_f_c, recon_loss_f_s, z_decoded, z_log_det = model.forward(
      inputs, labels, bert_inputs=sentence_embeddings, layer=2)
  z_star_list.extend(z_star.detach().cpu().numpy())
  labels_list.extend([dataset.idx2label(int(label))
                      for label in labels.tolist()])

  # to get the sentence
  seq_len, mbsize = inputs.size()
  num_range = torch.arange(
      0, seq_len).type(torch.ByteTensor).repeat(mbsize, 1).transpose(0, 1)
  num_range = num_range.cuda() if model.gpu else num_range
  lengths = torch.mul((inputs.data == model.EOS_IDX), num_range).sum(dim=0)
  orig_sentences = [dataset.idxs2sentence(
      x) for x in inputs.transpose(0, 1).data.tolist()]
  for i in range(len(lengths)):
    inputs_list.append(orig_sentences[i].split()[1:lengths[i]])


z_star_ = np.array(z_star_list)
# labels_list = torch.stack(labels_list, dim=0).detach().cpu().numpy()

m = -1
max_senti = 6.880212
min_senti = -6.1082063

num_samples = 1000
num_per_sample = 100
neg_sentences = []
pos_sentences = []


delta = (max_senti-min_senti)/19
senti_values = [min_senti+i*delta for i in range(20)]
print('all senti values')
print(senti_values)
print()
#x = senti_values[:5] + senti_values[-5:]
generated_sentences = [[] for _ in range(20)]

# posterior
# for i in range(num_samples):
for i in trange(num_samples):
  idx = i
  z_star_c_numpy = [z_star_[idx]]
  for s_idx in range(20):
    z_star_c_numpy[0][m] = senti_values[s_idx]
    z_star_c = torch.FloatTensor(z_star_c_numpy)
    z_star_c = z_star_c.cuda() if args.gpu else z_star_c
    t1 = time.time()
    z_decoded, z_log_det = model.forward_decoder_z(z_star_c)
    for j in range(num_per_sample):
      sample_idxs = model.sample_sentence(z_decoded, temp=0.2)
      sampled_sentence = dataset.idxs2sentence(sample_idxs)
      # print("prop:", p, "z:", z_star_c[0][m], sampled_sentence)
      generated_sentences[s_idx].append(sampled_sentence)


if not args.output_dir:
  output_dir = args.save_dir
else:
  output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'original.txt'), 'w') as f:
  for o in inputs_list[num_samples]:
    for j in range(num_per_sample):
      f.write(' '.join(o))
      f.write('\n')

with open(os.path.join(output_dir, 'labels.txt'), 'w') as f:
  for label in labels_list[num_samples]:
    for j in range(num_per_sample):
      f.write(str(label))
      f.write('\n')

print('saving to:', output_dir)
with open(output_dir+'/senti_values', 'w') as f:
  for s in senti_values:
    f.write(str(s))
    f.write('\n')

for s_idx in range(20):
  with open(output_dir+'/post.'+str(19-s_idx), 'w') as f:
    for sent in generated_sentences[s_idx]:
      f.write(sent)
      f.write('\n')
