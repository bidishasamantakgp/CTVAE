import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.optim as optim
import numpy as np
import copy
from math import ceil

from torch.autograd import Variable
from utils import *
from dataset import *
#from torch.distributions import MultivariateNormal
from model import RNN_VAE
# from BERT import *
from flows import *
import argparse
from tqdm import trange

parser = argparse.ArgumentParser(
    description='Conditional Text Generation: Train VAE as in Bowman, 2016, with c ~ p(c)'
)
parser.add_argument('--data', default='family', help='name of dataset')
parser.add_argument('--bert', default='../BERT',
                    help='name of bert embedding folder')
parser.add_argument('--gpu', default=True, action='store_true',
                    help='whether to run in the GPU')
parser.add_argument('--save', default=True, action='store_true',
                    help='whether to save model or not')
parser.add_argument('--save_dir', default='models_yelp', help='model_dir')
parser.add_argument('--with_bert', default=False, action='store_true',
                    help='if with bert sentence encodings')

parser.add_argument("--flows", default=2, type=int)
#parser.add_argument("--flow", default="MAF", type=str)
parser.add_argument("--flow", default="RealNVP", type=str)
parser.add_argument('--h_dim', default=200, type=int, help='encoder convertion')
parser.add_argument('--h_dim_1', default=100, type=int,
                    help='hidden dimension of fetaure layer')
parser.add_argument('--z_dim', default=256, type=int,
                    help='whether to save model or not')
parser.add_argument('--z_dim_1', default=256, type=int,
                    help='whether to save model or not')
parser.add_argument('--f_emb_dim', default=2, type=int,
                    help='whether to save model or not')
parser.add_argument('--feature_dim', default=32, type=int,
                    help='whether to save model or not')
parser.add_argument('--p_word_dropout', default=0.3, type=float,
                    help='word dropout rate')

parser.add_argument('--beta1_D', default=0.9, type=float,
                    help='beta1 parameter of the Adam optimizer for the discriminator')
parser.add_argument('--beta2_D', default=0.99, type=float,
                    help='beta2 parameter of the Adam optimizer for the discriminator')
parser.add_argument('--mbsize', default=32, type=int)
parser.add_argument('--epochs', default=20, type=int)

args = parser.parse_args()
with_bert = args.with_bert

lr = 0.001
lr_f = 0.001
lr_D = 0.0001

# lr_decay_every = 1000000
mb_size = 32

device = 'cuda' if args.gpu else 'cpu'

if args.data.lower() == 'imdb':
  print('using data imdb')
  #dataset = Imdb_Dataset(mbsize=mb_size, label = 2, repeat=False, shuffle=False)
  dataset = Imdb_Dataset(mbsize=mb_size, label = 2)
  MODEL_DIR = 'models_imdb/'
  train_size = 708929
  val_size = 4000
  #dataset = Imdb_Dataset(mbsize=mb_size)
  #MODEL_DIR = 'models_imdb/'

if args.data.lower() == 'yelp':
  print('using data yelp')
  dataset = Yelp_Dataset(mbsize=mb_size)
  MODEL_DIR = 'models_yelp/'
  train_size = 443248
  val_size = 4000

if args.data.lower() == 'family':
  #dataset = Amazon_Dataset(mbsize=mb_size)
  #MODEL_DIR = 'models_amazon/'
  print('using data music')
  dataset = Family_Dataset(mbsize=mb_size)
  MODEL_DIR = 'models_family/'
  train_size = 103934
  #train_size = 35800
  val_size = 2351


#dataset = Yelp_datase()

n_epochs = args.epochs
# n_iter = (n_epochs*train_size)//mb_size
# log_interval = (train_size//mb_size)//10  # 10 times per epoch
# save_interval = n_iter//4

print("dataset loaded.")
print("Dataset label:", dataset.n_vocab, dataset.LABEL.vocab.itos, dataset.LABEL.vocab.stoi, len(dataset.INDEX.vocab.itos))
#exit()

'''
label_list = list()
for batch in dataset.train_iter:
#for batch in dataset.train_iter:
  inputs, labels, indices = batch.text, batch.label, batch.index
  label_list.extend(labels)

np.savetxt('test_labels.txt',torch.stack(label_list).detach().cpu().numpy())
'''
#exit()
# bert_word_embedding = torch.load(
#     args.bert + "/" + args.data.lower() + '_train_word.pt')
# print('bert word embedding size',bert_word_embedding.size())
if with_bert:
  bert_sentence_embedding = torch.load("../BERT/family_train_sentence.pt")
  print("Bert sentence encoding size:", bert_sentence_embedding.size())
  assert bert_sentence_embedding.size()[0] == train_size

# n_batches = (bert_sentence_embedding.size()[0]) // mb_size + 1
n_batches = ceil(train_size/mb_size)
print("n_batchs:", n_batches)
n_iter = n_batches * n_epochs
log_interval = n_batches//3  # n times per epoch

thresh_iters = n_batches*60  ## 10 epochs of lower level training

flow = eval(args.flow)
flows = [flow(dim=args.z_dim) for _ in range(args.flows)]

model = RNN_VAE(args, dataset.n_vocab,
            flows,    
            pretrained_embeddings=dataset.get_vocab_vectors(),
                freeze_embeddings=True,
                gpu=args.gpu,
                with_bert_encodings=with_bert
                )


#model.load_state_dict(torch.load(args.save_dir+"/vae_working.bin"))
#model.load_state_dict(torch.load("ny_flow_gab1/vae55577.bin"))
#model.load_state_dict(torch.load("imdb_flow_3_feature/vae206780.bin"))
#model.load_state_dict(torch.load("imdb_flow_3_feature/vae.bin"))
#model.load_state_dict(torch.load("imdb_flow_3_feature2/vae.bin"))
'''
#for name, param in model.vae_params:
for name, param in model.named_parameters():
    if param.requires_grad:
         print(name)

for p in model.lower_level_params:
    print("lower:", p)

for p in model.upper_level_params:
    print("upper:", p)
'''
#exit()

#print(model.parameters())
# model_D = Discriminator(args.z_dim_1).to(device)


def main():
  min_loss = 5000
  # Annealing for KL term
  #kld_start_inc = 13852 * 5
  kld_start_inc = 0.2 * thresh_iters 
  kld_end_inc = 0.5*thresh_iters
  kld_weight = 0.1
  kld_max = 10.0
  #kld_max = 10
  # kld_inc = (kld_max - kld_weight) / (25000 - kld_start_inc)
  kld_inc = (kld_max - kld_weight) / (kld_end_inc - kld_start_inc + 1)
  #kld_inc = (kld_max - kld_weight) / (250 - kld_start_inc)


  kld_start_inc_f = thresh_iters
  kld_end_inc_f = kld_start_inc_f + 0.7*(n_iter-kld_start_inc_f)
  kld_weight_f = 0.001
  kld_max_f = 0.1
  # kld_inc_f = (kld_max_f - kld_weight_f) / (30000 - kld_start_inc_f)
  kld_inc_f = (kld_max_f - kld_weight_f) / (kld_end_inc_f - kld_start_inc_f + 1)
  #kld_inc_f = (kld_max_f - kld_weight_f) / (300 - kld_start_inc_f)

  loss_start_inc_f = thresh_iters
  loss_end_inc_f = loss_start_inc_f + 0.6*(n_iter-loss_start_inc_f)
  loss_weight_f = 0.01
  loss_max_f = 1
  # loss_inc_f = (loss_max_f - loss_weight_f) / (30000 - loss_start_inc_f)
  loss_inc_f = (loss_max_f - loss_weight_f) / (loss_end_inc_f - loss_start_inc_f+1)

  assert kld_end_inc<n_iter
  #assert kld_end_inc_f<n_iter
  #assert loss_end_inc_f<n_iter

  loss_f_s_weight = 1000
  recon_loss_weight = 500
  assert loss_f_s_weight>1

  gamma_start_inc_f = 400
  gamma_weight_f = 0
  gamma_max_f = 10
  gamma_inc_f = (gamma_max_f - gamma_weight_f) / (n_iter - gamma_start_inc_f)

  trainer = optim.Adam(model.parameters(), lr=lr,
                       betas=(args.beta1_D, args.beta2_D))
  sentence_trainer = optim.Adam(model.lower_level_params, lr=lr,
                       betas=(args.beta1_D, args.beta2_D))
  
  flow_trainer = optim.Adam(model.upper_level_params, lr=lr_f,
                       betas=(args.beta1_D, args.beta2_D))
  alpha_weight_f = 1.0
  dis = True

  for it in trange(n_iter):
    inputs, labels, indices = dataset.next_batch(args.gpu)
    # print('First:', dataset.idxs2sentence(inputs[:,0]))
    # print('First label:', dataset.idx2label(labels[0]))
    # print('First index:', indices[0])
    #print(labels)
    labels = [dataset.LABEL.vocab.itos[l] for l in labels]
    labels = torch.LongTensor(labels).cuda() if args.gpu else  torch.LongTensor(labels)
    #print(labels)
    #exit()
    
    if with_bert:
      sentence_embeddings = torch.stack(
          [bert_sentence_embedding[idx] for idx in indices])
      if args.gpu:
        sentence_embeddings = sentence_embeddings.cuda()
    else:
        sentence_embeddings = None
    #labels_new = [dataset.LABEL.vocab.itos[x] for x in labels.detach().cpu().numpy()]
    #print(labels_new)
    #labels_new = torch.LongTensor(labels_new).cuda() if args.gpu else  torch.LongTensor(labels_new)
    
    if it<thresh_iters:
          mu, logvar, z,recon_loss, kl_loss_z_prior = model.forward(inputs, labels, bert_inputs=sentence_embeddings, layer=1)
          lower_level_loss = recon_loss_weight * recon_loss + kld_weight * kl_loss_z_prior
    else:
          z, z_star, z_star_prior_logprob, z_star_log_det, recon_loss_f_c, recon_loss_f_s, z_decoded, z_log_det,kl_z_star = model.forward(inputs, labels, bert_inputs=sentence_embeddings, layer=2)
          #print(kl_z_star)
          z_star_prior_logprob = torch.mean(z_star_prior_logprob)
          z_star_log_det = torch.mean(z_star_log_det)
          logprob_z_star =  10 * z_star_prior_logprob + z_star_log_det
          loss_z_star = -logprob_z_star + kld_weight_f * kl_z_star
          loss_f = loss_f_s_weight * recon_loss_f_s
          upper_level_loss = loss_f + loss_z_star
    '''    
    mu, logvar, z, z_star, z_star_prior_logprob, z_star_log_det, recon_loss, recon_loss_f_c, \
            recon_loss_f_s, kl_loss_z_prior,  z_decoded, z_log_det = \
            model.forward(inputs, labels, bert_inputs=sentence_embeddings)

    z_star_prior_logprob = torch.mean(z_star_prior_logprob)
    z_star_log_det = torch.mean(z_star_log_det)
    logprob_z_star = kld_weight_f * z_star_prior_logprob + z_star_log_det
    #logprob_z_star = z_star_prior_logprob + z_star_log_det
    loss_z_star = -logprob_z_star
    loss_z = recon_loss + kld_weight * kl_loss_z_prior
    loss_f = loss_f_s_weight * recon_loss_f_s #- recon_loss_f_c
    #loss_z_star = kld_weight_f * kl_loss_z_star  # + loss_weight_f *kl_loss_z

    #nll_local = -loglikelihood(mu, logvar, z.detach())
    #loss_z_star = nll_z_star + kld_weight_f * kl_loss_z_star
    # D_z = model_D.forward(z_star.detach())

    vae_loss_total = loss_z + loss_f + loss_z_star
    lower_level_loss = recon_loss_weight * recon_loss + kld_weight * kl_loss_z_prior
    upper_level_loss = loss_f + loss_z_star
    
    #print("f_alpha:", f_alpha.detach().numpy())
    # if it % log_interval == 0:
    #   print("z_:", z.detach()[:10, :10])
    #   print("z_star:", z_star.detach()[:10, :10])
    #   #print("f_alpha:", f_alpha.detach().numpy()[:10,:])
    #   print("labels:", labels)
    #   print("nll_local", nll_local)
    #   print("mu:", mu.detach()[:10, :10])
    #   print("mu_decoder:", d_mu.detach()[:10, :10])

    '''
    if it > kld_start_inc_f and kld_weight_f < kld_max_f:
      kld_weight_f += kld_inc_f

    if it > kld_start_inc and kld_weight < kld_max:
      kld_weight += kld_inc

    if it > loss_start_inc_f and loss_weight_f < loss_max_f:
      loss_weight_f += loss_inc_f

    if it > gamma_start_inc_f and gamma_weight_f < gamma_max_f:
      gamma_weight_f += gamma_inc_f

    if it<thresh_iters:
      lower_level_loss.backward()
      grad_norm = torch.nn.utils.clip_grad_value_(model.lower_level_params, 50)
      sentence_trainer.step()
      sentence_trainer.zero_grad()
    else:
      upper_level_loss.backward()
      grad_norm = torch.nn.utils.clip_grad_value_(model.upper_level_params, 50)
      flow_trainer.step()
      flow_trainer.zero_grad()
      #vae_loss_total.backward()
      #grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), 50)
      #trainer.step()
      #trainer.zero_grad()
    dis = False
    #print("z:",z[0])
    #print("z_decoded:", z_decoded[0])
    #print("z_star:", z_star[0])
    if (it % log_interval) == 0 or it == (n_iter-1):
       if it>=thresh_iters and loss_f.data < min_loss:
             min_loss = loss_f.data
             save_model(min_loss)
       save_model()
       model.eval()
       if it <  thresh_iters:
         input_sent = dataset.idxs2sentence(inputs.t()[0])
         print("Input:", input_sent)
         sample_idxs = model.sample_sentence(z[0], temp=0.1)
         sample_sent = dataset.idxs2sentence(sample_idxs)
         print("Posterior:", sample_sent)
         print('Iter-{}; Sentence Loss: {:.4f}; Recon: {:.4f}; KL_z_prior: {:.10f}; KLD_weight: {:.10f}'.format(
            it, lower_level_loss.data, recon_loss.data, kl_loss_z_prior.data, kld_weight))
       else:
         print("z:",z[0])
         print("z_decoded:", z_decoded[0])
         print("z_star:", z_star[0])
 
         input_sent = dataset.idxs2sentence(inputs.t()[0])
         print("Input:", input_sent)
 
         sample_idxs = model.sample_sentence(z[0], temp=0.1)
         sample_sent = dataset.idxs2sentence(sample_idxs)
         print("Posterior:", sample_sent)
 
         sample_idxs = model.sample_sentence(z_decoded[0], temp=0.1)
         sample_sent = dataset.idxs2sentence(sample_idxs)
         print("Posterior:", sample_sent)
 
         #print('Iter-{}; Recon: {:.4f}; KL_z_prior: {:.10f}; KLD_weight: {:.10f}'.format(
         #  it, recon_loss.data, kl_loss_z_prior.data, kld_weight))
         print('Iter-{}; Feature Loss: Recon: {:.4f}; Weight:{:.4f}; KL:{:.4f}'.format(it, loss_f.data, loss_f_s_weight, kl_z_star.data))
         print('Iter-{}; z_star_prio_logprob*: {:.10f}; KLD_log_det: {:.10f}; KLD_weight_f: {:.10f};'.format(it, z_star_prior_logprob , z_star_log_det, kld_weight_f)) 

    model.train()
  
  save_model()


def save_model(min_loss=0):
  if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
  if min_loss != 0:
      torch.save(model.state_dict(), args.save_dir+'/vae'+str(min_loss)+'.bin')
  else:
    torch.save(model.state_dict(), args.save_dir+'/vae.bin')
  # torch.save(model_D.state_dict(), args.save_dir+'/vae_d.bin')


if __name__ == '__main__':
  try:
    main()
  except KeyboardInterrupt:
    if args.save:
      save_model()
    exit(0)

  if args.save:
    save_model()
