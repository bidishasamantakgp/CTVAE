import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import chain
from utils import *
import math
from torch.distributions import MultivariateNormal, Normal


class RNN_VAE(nn.Module):

  def __init__(self, args, n_vocab,  flows, encoder_bert=None, pretrained_embeddings=None, freeze_embeddings=False, gpu=False,
  max_sent_len=30,
  with_bert_encodings=False):
    super(RNN_VAE, self).__init__()

    self.UNK_IDX = 0
    self.PAD_IDX = 1
    self.START_IDX = 2
    self.EOS_IDX = 3
    self.MAX_SENT_LEN = max_sent_len

    self.with_bert_encodings=with_bert_encodings
    
    self.n_vocab = n_vocab
    self.h_dim = args.h_dim
    self.h_dim_1 = args.h_dim_1
    self.z_dim = args.z_dim
    self.z_dim_1 = args.z_dim_1
    self.p_word_dropout = args.p_word_dropout
    #self.c_dim = args.c_dim
    self.f_emb_dim = args.f_emb_dim
    self.feature_dim = args.feature_dim
    self.gpu = gpu
    self.emb_dim_enc = 768
    self.mb_size = args.mbsize
    self.flows = nn.ModuleList(flows)
    #self.flows = flows
    """
      Word embeddings layer
    """
    mu = np.zeros(self.z_dim)
    #mu[-1] = 0.0
    sigma = np.ones(self.z_dim)
    #sigma[-1] = 5.0
    #mu[-1] = 0.5
    #sigma[-1] = 0.5
    mu = torch.FloatTensor(mu)
    sigma = torch.FloatTensor(sigma)

    #sigma = torch.eye(self.z_dim)
    
    mu = mu.cuda() if gpu else mu
    sigma = sigma.cuda() if gpu else sigma
    self.prior = Normal(mu, sigma)
    #MultivariateNormal(mu, sigma)
    #self.prior = Normal() 
    if pretrained_embeddings is None:
      self.emb_dim = self.h_dim
      self.word_emb = nn.Embedding(n_vocab, args.h_dim, self.PAD_IDX)

    else:
      self.emb_dim = pretrained_embeddings.size(1)
      self.word_emb = nn.Embedding(n_vocab, self.emb_dim, self.PAD_IDX)

      # Set pretrained embeddings x
      self.word_emb.weight.data.copy_(pretrained_embeddings)

      if freeze_embeddings:
        self.word_emb.weight.requires_grad = False

    """
        Encoder is GRU with FC layers connected to last hidden unit
    """
    # Encoder sentence sentence to z
    if self.with_bert_encodings:
      self.encoder = nn.Linear(self.emb_dim_enc, self.h_dim)
    else:
      self.encoder = nn.GRU(self.emb_dim, self.h_dim)
    self.q_stat = nn.Linear(self.h_dim, 2*self.z_dim)

    # self.q_stat = nn.Linear(self.h_dim, 2 * self.z_dim)

    # Encoder conversion z -> z*

    # Feature tuning sentiment z* -> sentiment
    #self.q_f_alpha = nn.Sequential(nn.Linear(1, self.f_emb_dim), nn.Softplus())
    self.q_f_alpha = nn.Linear(self.z_dim_1-1, self.f_emb_dim)
    self.q_f_alpha_1 = nn.Linear(1, self.f_emb_dim)

    # Decode z* --> z

    # Decoder sentence z to sentence
    self.decoder = nn.GRU(self.emb_dim + self.z_dim, self.z_dim, dropout=0.3)
    self.decoder_fc = nn.Linear(self.z_dim, n_vocab)

    self.encoder_params = chain(
        self.encoder.parameters(),
        self.q_stat.parameters(),
    )

    self.encoder_f_params = chain(
        # self.encoder_feature.parameters(),
        self.q_f_alpha.parameters(),
        self.q_f_alpha_1.parameters()
    )

    self.decoder_params = chain(
        self.decoder.parameters(), self.decoder_fc.parameters()
    )


    self.vae_params = chain(
        self.word_emb.parameters(), self.encoder_params, self.encoder_f_params, self.decoder_params
    )

    self.lower_level_params = chain(
      self.word_emb.parameters(),
      self.encoder.parameters(), self.q_stat.parameters(),
      self.decoder.parameters(), self.decoder_fc.parameters()
    )
    self.upper_level_params = chain(
            self.flows.parameters(), self.q_f_alpha.parameters(), self.q_f_alpha_1.parameters())
    #self.lower_level_params = filter(lambda p: p.requires_grad,
    #                                 self.lower_level_params)

    #self.vae_params = filter(lambda p: p.requires_grad, self.vae_params)

    """
        Use GPU if set
        """
    if self.gpu:
      self.cuda()

  def forward_encoder(self, inputs, lengths, bert_inputs=None):
    """
    Inputs is batch of sentences: seq_len x mbsize
    """
    if self.with_bert_encodings:
      return self.forward_encoder_embed(bert_inputs, None)

    sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
    _, reversed_idx = torch.sort(sorted_idx)
    inputs = inputs[:, sorted_idx]
    inputs = self.word_emb(inputs)
    packed_inputs = nn.utils.rnn.pack_padded_sequence(
        inputs, sorted_lengths.tolist(), batch_first=False)
    return self.forward_encoder_embed(packed_inputs, reversed_idx)

  def forward_encoder_embed(self, inputs, reversed_idx):
    """
    Inputs is embeddings of: seq_len x mbsize x emb_dim
    """
    if self.with_bert_encodings:
      mb_size = inputs.size()[0]
      h = self.encoder(inputs)
    else:
      _, h = self.encoder(inputs, None)
      h = h[:, reversed_idx, :]

    # Forward to latent
    h = h.view(-1, self.h_dim)
    stats = self.q_stat(h)
    mu, logvar = stats[:, :self.z_dim], stats[:, self.z_dim:]
    return mu, logvar

  def forward_encoder_z_star(self, z):
        m, _ = z.shape
        log_det = torch.zeros(m)
        log_det = log_det.cuda() if self.gpu else log_det 
        for flow in self.flows:
            z, ld, mu, sigma = flow.forward(z)
            log_det += ld
        print("shape z:", z.size())
        z, prior_logprob = z, self.prior.log_prob(z)
        return z, prior_logprob, log_det, mu, sigma


  def forward_decoder_z(self, zstar):
        m, _ = zstar.shape
        z = zstar
        log_det = torch.zeros(m)
        log_det = log_det.cuda() if self.gpu else log_det
        for flow in self.flows[::-1]:
            z, ld = flow.inverse(z)
            log_det += ld
        x = z
        return x, log_det

  def sample_z(self, n_samples):
        z = self.prior.sample((n_samples,))
        x, _ = self.forward_decoder_z(z.detach())
        return x

  def sample_z_gaussian(self, mu, logvar):
    """
    Reparameterization trick: z = mu + std*eps; eps ~ N(0, I)
    """
    dim = mu.size()[-1]
    eps = Variable(torch.randn(dim))
    eps = eps.cuda() if self.gpu else eps
    return mu + torch.exp(logvar/2) * eps
  
  def forward_encoder_feature(self, z_star):
    alpha1 = self.q_f_alpha(z_star[:, :-1])
    #alpha2 = self.q_f_alpha_1(z_star[:, -1:])
    #return alpha1, alpha2
    mb_size = z_star.shape[0]
    #print("z_star shape", z_star[:,-1].shape)
    #scale = torch.FloatTensor([[-0.5,0.5]])
    scale = torch.FloatTensor([[-1.0, 1.0]])
    scale = scale.cuda() if self.gpu else scale
    return alpha1, torch.matmul(z_star[:,-1:], scale)
    # return torch.distributions.Dirichlet(alpha)



  def sample_z_star_prior(self, mbsize, feature):
    """
    Sample z ~ p(z) = N(0, I)
    """
    z = Variable(torch.randn(mbsize, self.z_dim_1 - 1))
    z_cat = torch.cat((z, feature), 1)
    z_cat = z_cat.cuda() if self.gpu else z_cat
    return z_cat

  def sample_z_prior(self, mbsize):
    """
    Sample z ~ p(z) = N(0, I)
    """
    z = Variable(torch.randn(mbsize, self.z_dim))
    z = z.cuda() if self.gpu else z
    return z

  def sample_z_star_control(self, feature):
    self.eval()
    mb_size, _ = feature.size()
    mu, logvar = self.forward_estimate_z_star(feature)
    z_feature = self.sample_z(mu, logvar)
    print("z_feature:", z_feature)
    # print(z_feature.size())
    z_star = self.sample_z_star_prior(mb_size, z_feature)
    print("z_star", z_star)
    mu, logvar = self.forward_decoder_z(z_star)
    z = self.sample_z(mu, logvar)
    # self.train()
    return z

  def sample_f_prior(self, mbsize):
    """
    Sample c ~ p(c) = Cat([0.5, 0.5])
    """
    f = Variable(
        torch.from_numpy(np.random.multinomial(
            1, [0.5, 0.5], mbsize).astype('float32'))
    )
    f = f.cuda() if self.gpu else f
    return f

  def sample_f(self, alpha):
    """
    Sample c ~ p(c) = Cat([0.5, 0.5])
    """

    f = F.softmax(alpha, dim=-1)
    #_, y = gumbel_softmax(f, temperature = 0.1)
    # Variable(torch.distributions.categorical.Categorical(alpha))
    # f = Variable(
    #    torch.from_numpy(np.random.multinomial(1, alpha.numpy(), ).astype('float32'))
    # )
    f = f.cuda() if self.gpu else f
    #y = y.cuda() if self.gpu else y
    # return f, y
    return f

  def forward_decoder(self, inputs, z, lengths):
    """
    Inputs must be embeddings: seq_len x mbsize
    """
    dec_inputs = self.word_dropout(inputs)

    sorted_lengths, sorted_idx = torch.sort(lengths, descending=True)
    _, reversed_idx = torch.sort(sorted_idx)

    # Forward
    seq_len = dec_inputs.size(0)

    # 1 x mbsize x (z_dim+c_dim)
    init_h = z.unsqueeze(0)
    #torch.cat([z.unsqueeze(0), c.unsqueeze(0)], dim=2)
    inputs_emb = self.word_emb(dec_inputs)  # seq_len x mbsize x emb_dim
    inputs_emb = torch.cat([inputs_emb, init_h.repeat(seq_len, 1, 1)], 2)

    inputs_emb = inputs_emb[:, sorted_idx, :]
    packed_inputs_emb = nn.utils.rnn.pack_padded_sequence(
        inputs_emb, sorted_lengths.tolist(), batch_first=False)

    outputs, _ = self.decoder(packed_inputs_emb, init_h)
    outputs = nn.utils.rnn.pad_packed_sequence(outputs)[0][:, reversed_idx, :]

    seq_len, mbsize, _ = outputs.size()

    outputs = outputs.view(seq_len*mbsize, -1)
    y = self.decoder_fc(outputs)
    y = y.view(seq_len, mbsize, self.n_vocab)

    return y

  def forward_decoder_feature(self, z):
    return self.decoder_feature(z)

  def forward_discriminator(self, inputs):
    """
    Inputs is batch of sentences: mbsize x seq_len
    """
    inputs = self.word_emb(inputs)
    return self.forward_discriminator_embed(inputs)

  def forward_discriminator_embed(self, inputs):
    """
    Inputs must be embeddings: mbsize x seq_len x emb_dim
    """
    inputs = inputs.unsqueeze(1)  # mbsize x 1 x seq_len x emb_dim

    x3 = F.relu(self.conv3(inputs)).squeeze()
    x4 = F.relu(self.conv4(inputs)).squeeze()
    x5 = F.relu(self.conv5(inputs)).squeeze()

    # Max-over-time-pool
    x3 = F.max_pool1d(x3, x3.size(2)).squeeze()
    x4 = F.max_pool1d(x4, x4.size(2)).squeeze()
    x5 = F.max_pool1d(x5, x5.size(2)).squeeze()

    x = torch.cat([x3, x4, x5], dim=1)

    y = self.disc_fc(x)

    return y

  def forward(self, sentence, labels, bert_inputs=None, no_decode=False, layer = 1):
    """
    Params:
    -------
    sentence: sequence of word indices.

    Returns:
    --------
    recon_loss: reconstruction loss of VAE.
    kl_loss: KL-div loss of VAE.
    """
    self.train()

    seq_len, mbsize = sentence.size()
    num_range = torch.arange(
        0, seq_len).type(torch.ByteTensor).repeat(mbsize, 1).transpose(0, 1)
    num_range = num_range.cuda() if self.gpu else num_range
    lengths = torch.mul((sentence.data == self.EOS_IDX), num_range).sum(dim=0)


    enc_inputs = sentence[1:]
    dec_inputs = sentence
    dec_targets = sentence[1:]
    dec_targets_f = labels

    # snetence -> z
    mu, logvar = self.forward_encoder(enc_inputs, lengths, bert_inputs=bert_inputs)
    z = self.sample_z_gaussian(mu, logvar)
    if layer == 1:
         y = self.forward_decoder(dec_inputs, z, lengths)
         y_seq_len = y.size(0)

         recon_loss = F.cross_entropy(
              y.view(-1, self.n_vocab), dec_targets[:y_seq_len].view(-1), size_average=False, ignore_index=self.PAD_IDX
         )/mbsize

         kl_loss_z_prior = torch.mean(
          0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))


         #if layer == 1:
         return mu, logvar, z,recon_loss, kl_loss_z_prior
    else:
         # z-> z*
         z_star, z_star_prior_logprob, z_star_log_det,mu_star, sigma_star = self.forward_encoder_z_star(z)
 
 
         # z* -> sentiment
         #f_alpha_c, f_alpha_s = self.forward_encoder_feature(z_star)
         f_alpha_c, f_alpha_s = self.forward_encoder_feature(z_star)
 
         # sentiment -> z* estimate
         # d_mu_star, d_logvar_star = self.forward_estimate_z_star(f_alpha_c)
 
         # z* -> z
 
         z_decoded, z_log_det = self.forward_decoder_z(z_star)
 
         # z -> sentence
         #y = self.forward_decoder(dec_inputs, z, lengths)
         #y_seq_len = y.size(0)
 
 
 
         #recon_loss = F.cross_entropy(
         #    y.view(-1, self.n_vocab), dec_targets[:y_seq_len].view(-1), size_average=False, ignore_index=self.PAD_IDX
         #)/mbsize
 
         #recon_loss_f_c = F.cross_entropy(
         #    f_alpha_c.view(-1, self.f_emb_dim), dec_targets_f.view(-1), size_average=False)/mbsize
 
         recon_loss_f_s = F.cross_entropy(
             f_alpha_s.view(-1, 2), dec_targets_f.view(-1), size_average=False)/mbsize
         
         recon_loss_f_c = 0
         logvar = 2 * sigma_star
         mu = mu_star
         kl_loss_z_prior = torch.mean(0.5 * torch.sum(torch.exp(logvar) + mu**2 - 1 - logvar, 1))
         
         return z, z_star, z_star_prior_logprob, z_star_log_det, recon_loss_f_c, recon_loss_f_s, z_decoded, z_log_det, kl_loss_z_prior

  def generate_sentences(self, batch_size):
    """
    Generate sentences and corresponding z of (batch_size x max_sent_len)
    """
    samples = []
    cs = []

    for _ in range(batch_size):
      z = self.sample_z_prior(1)
      #c = self.sample_c_prior(1)
      samples.append(self.sample_sentence(z, raw=True))
      cs.append(c.long())

    X_gen = torch.cat(samples, dim=0)
    c_gen = torch.cat(cs, dim=0)

    return X_gen, c_gen

  def sample_sentence(self, z, raw=False, temp=1):
    """
    Sample single sentence from p(x|z,c) according to given temperature.
    `raw = True` means this returns sentence as in dataset which is useful
    to train discriminator. `False` means that this will return list of
    `word_idx` which is useful for evaluation.
    """
    self.eval()

    word = torch.LongTensor([self.START_IDX])
    word = word.cuda() if self.gpu else word
    word = Variable(word)  # '<start>'

    #z, c = z.view(1, 1, -1), c.view(1, 1, -1)
    z = z.view(1, 1, -1)
    h = z
    #torch.cat([z, c], dim=2)

    if not isinstance(h, Variable):
      h = Variable(h)

    outputs = []

    if raw:
      outputs.append(self.START_IDX)

    for i in range(self.MAX_SENT_LEN):
      emb = self.word_emb(word).view(1, 1, -1)
      emb = torch.cat([emb, z], 2)

      output, h = self.decoder(emb, h)
      y = self.decoder_fc(output).view(-1)
      y = F.softmax(y/temp, dim=0)

      idx = torch.multinomial(y, 1)

      word = Variable(torch.LongTensor([int(idx)]))
      word = word.cuda() if self.gpu else word
      idx = int(idx)

      if not raw and (idx == self.EOS_IDX or idx == self.PAD_IDX):
        break

      outputs.append(idx)

    # Back to default state: train
    self.train()

    if raw:
      outputs = Variable(torch.LongTensor(outputs)).unsqueeze(0)
      return outputs.cuda() if self.gpu else outputs
    else:
      return outputs

  def generate_soft_embed(self, mbsize, temp=1):
    """
    Generate soft embeddings of (mbsize x emb_dim) along with target z
    and c for each row (mbsize x {z_dim, c_dim})
    """
    samples = []
    targets_c = []
    targets_z = []

    for _ in range(mbsize):
      z = self.sample_z_prior(1)
      c = self.sample_c_prior(1)

      samples.append(self.sample_soft_embed(z, c, temp=1))
      targets_z.append(z)
      targets_c.append(c)

    X_gen = torch.cat(samples, dim=0)
    targets_z = torch.cat(targets_z, dim=0)
    _, targets_c = torch.cat(targets_c, dim=0).max(dim=1)

    return X_gen, targets_z, targets_c

  def sample_soft_embed(self, z, c, temp=1):
    """
    Sample single soft embedded sentence from p(x|z,c) and temperature.
    Soft embeddings are calculated as weighted average of word_emb
    according to p(x|z,c).
    """
    self.eval()

    z, c = z.view(1, 1, -1), c.view(1, 1, -1)

    word = torch.LongTensor([self.START_IDX])
    word = word.cuda() if self.gpu else word
    word = Variable(word)  # '<start>'
    emb = self.word_emb(word).view(1, 1, -1)
    emb = torch.cat([emb, z, c], 2)

    h = torch.cat([z, c], dim=2)

    if not isinstance(h, Variable):
      h = Variable(h)

    outputs = [self.word_emb(word).view(1, -1)]

    for i in range(self.MAX_SENT_LEN):
      output, h = self.decoder(emb, h)
      o = self.decoder_fc(output).view(-1)

      # Sample softmax with temperature
      y = F.softmax(o / temp, dim=0)

      # Take expectation of embedding given output prob -> soft embedding
      # <y, w> = 1 x n_vocab * n_vocab x emb_dim
      emb = y.unsqueeze(0) @ self.word_emb.weight
      emb = emb.view(1, 1, -1)

      # Save resulting soft embedding
      outputs.append(emb.view(1, -1))

      # Append with z and c for the next input
      emb = torch.cat([emb, z], 2)

    # 1 x 16 x emb_dim
    outputs = torch.cat(outputs, dim=0).unsqueeze(0)

    # Back to default state: train
    self.train()

    return outputs.cuda() if self.gpu else outputs

  def word_dropout(self, inputs):
    """
    Do word dropout: with prob `p_word_dropout`, set the word to '<unk>'.
    """
    if isinstance(inputs, Variable):
      data = inputs.data.clone()
    else:
      data = inputs.clone()

    # Sample masks: elems with val 1 will be set to <unk>
    mask = torch.from_numpy(
        np.random.binomial(1, p=self.p_word_dropout, size=tuple(data.size()))
                 .astype('bool')
    )
                #  .astype('uint8')
    # )

    if self.gpu:
      mask = mask.cuda()

    # Set to <unk>
    data[mask] = self.UNK_IDX

    return Variable(data)
