from torchtext import data, datasets
from torchtext.vocab import GloVe
from dataset_yelp import Yelp
from dataset_amazon import Amazon
from dataset_gab import Gab
from dataset_music import Music
from dataset_family import Family


class Yelp_Dataset:

  def __init__(self, emb_dim=300, mbsize=32, shuffle=True, repeat=True):
    self.TEXT = data.Field(init_token='<start>', eos_token='<eos>',
                           lower=True, tokenize='spacy', fix_length=15+2)
    self.LABEL = data.Field(sequential=False, unk_token=None)
    self.INDEX = data.Field(sequential=False, unk_token=None)


    train, val, test = Yelp.splits(self.TEXT, self.LABEL, self.INDEX
                                   )

    self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
    self.LABEL.build_vocab(train)
    self.INDEX.build_vocab(train)

    self.n_vocab = len(self.TEXT.vocab.itos)
    self.emb_dim = emb_dim

    self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=mbsize, device=-1,
        shuffle=shuffle, repeat=repeat
    )
    self.train_iter = iter(self.train_iter)
    self.val_iter = iter(self.val_iter)
    self.test_iter = iter(self.test_iter)

  def get_vocab_vectors(self):
    return self.TEXT.vocab.vectors

  def next_batch(self, gpu=False):
    batch = next(self.train_iter)

    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def next_validation_batch(self, gpu=False):
    batch = next(self.val_iter)

    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def idxs2sentence(self, idxs):
    return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

  def idx2label(self, idx):
    return self.LABEL.vocab.itos[idx]


class Music_Dataset:

  def __init__(self, emb_dim=300, mbsize=32, shuffle=True, repeat=True):
    
    self.TEXT = data.Field(init_token='<start>',
                          eos_token='<eos>', tokenize='spacy', fix_length = 25 + 2)
    self.LABEL = data.Field(sequential=False, unk_token=None)
    self.INDEX = data.Field(sequential=False, unk_token=None)


    train, val, test = Music.splits(
        self.TEXT, self.LABEL, self.INDEX
    )
    
    
    self.TEXT.build_vocab(train, min_freq = 2, vectors=GloVe('6B', dim=emb_dim))
    self.LABEL.build_vocab(train)
    self.INDEX.build_vocab(train)

    self.n_vocab = len(self.TEXT.vocab.itos)
    self.emb_dim = emb_dim
    
    self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=mbsize, device=-1,
        # shuffle=True, repeat=True
        shuffle=shuffle, repeat=shuffle
    )
    self.train_iter = iter(self.train_iter)
    self.val_iter = iter(self.val_iter)
    self.test_iter = iter(self.test_iter)

  def get_vocab_vectors(self):
    return self.TEXT.vocab.vectors

  def next_batch(self, gpu=False):
    batch = next(self.train_iter)
    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def next_validation_batch(self, gpu=False):
    batch = next(self.test_iter)

    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def idxs2sentence(self, idxs):
    return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

  def idx2label(self, idx):
    return self.LABEL.vocab.itos[idx]

class Family_Dataset:

  def __init__(self, emb_dim=300, mbsize=32, shuffle=True, repeat=True):
    self.TEXT = data.Field(init_token='<start>',
                           eos_token='<eos>', lower=True, tokenize='spacy', fix_length = 30 + 2)
    self.LABEL = data.Field(sequential=False, unk_token=None)
    self.INDEX = data.Field(sequential=False, unk_token=None)


    train, val, test = Family.splits(
        self.TEXT, self.LABEL, self.INDEX
    )

    self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
    self.LABEL.build_vocab(train)
    self.INDEX.build_vocab(train)

    self.n_vocab = len(self.TEXT.vocab.itos)
    self.emb_dim = emb_dim

    self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=mbsize, device=-1,
        # shuffle=True, repeat=True
        shuffle=shuffle, repeat=shuffle
    )
    self.train_iter = iter(self.train_iter)
    self.val_iter = iter(self.val_iter)
    self.test_iter = iter(self.test_iter)

  def get_vocab_vectors(self):
    return self.TEXT.vocab.vectors

  def next_batch(self, gpu=False):
    batch = next(self.train_iter)
    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def idxs2sentence(self, idxs):
    return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

  def idx2label(self, idx):
    return self.LABEL.vocab.itos[idx]

class Gab_Dataset:

  def __init__(self, emb_dim=300, mbsize=32, shuffle=True, repeat=True):
    self.TEXT = data.Field(init_token='<start>',
                           eos_token='<eos>', lower=True, tokenize='spacy', fix_length = 35 + 2)
    self.LABEL = data.Field(sequential=False, unk_token=None)
    self.INDEX = data.Field(sequential=False, unk_token=None)


    train, val, test = Gab.splits(
        self.TEXT, self.LABEL, self.INDEX
    )
    
    self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
    self.LABEL.build_vocab(train)
    self.INDEX.build_vocab(train)

    self.n_vocab = len(self.TEXT.vocab.itos)
    self.emb_dim = emb_dim

    self.train_iter, self.val_iter, self.test_iter = data.BucketIterator.splits(
        (train, val, test), batch_size=mbsize, device=-1,
        # shuffle=True, repeat=True
        shuffle=shuffle, repeat=shuffle
    )
    self.train_iter = iter(self.train_iter)
    self.val_iter = iter(self.val_iter)
    self.test_iter = iter(self.test_iter)

  def get_vocab_vectors(self):
    return self.TEXT.vocab.vectors

  def next_batch(self, gpu=False):
    batch = next(self.train_iter)

    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def next_validation_batch(self, gpu=False):
    batch = next(self.test_iter)

    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def idxs2sentence(self, idxs):
    return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

  def idx2label(self, idx):
    return self.LABEL.vocab.itos[idx]


class Amazon_Dataset:

  def __init__(self, emb_dim=300, mbsize=32, shuffle=True, repeat=True):
    self.TEXT = data.Field(init_token='<start>',
                           eos_token='<eos>', lower=True, tokenize='spacy')
    self.LABEL = data.Field(sequential=False, unk_token=None)
    self.INDEX = data.Field(sequential=False, unk_token=None)

    # Only take sentences with length <= 50
    # f = lambda ex: len(ex.text) <= 50 #and ex.label != 'neutral'

    train, val, test = Amazon.splits(
        self.TEXT, self.LABEL, self.INDEX
    )

    self.TEXT.build_vocab(train, vectors=GloVe('6B', dim=emb_dim))
    self.LABEL.build_vocab(train)
    self.INDEX.build_vocab(train)

    self.n_vocab = len(self.TEXT.vocab.itos)
    self.emb_dim = emb_dim

    self.train_iter, self.val_iter, _ = data.BucketIterator.splits(
        (train, val, test), batch_size=mbsize, device=-1,
        # shuffle=True, repeat=True
        shuffle=shuffle, repeat=shuffle
    )
    self.train_iter = iter(self.train_iter)
    self.val_iter = iter(self.val_iter)

  def get_vocab_vectors(self):
    return self.TEXT.vocab.vectors

  def next_batch(self, gpu=False):
    batch = next(self.train_iter)

    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def next_validation_batch(self, gpu=False):
    batch = next(self.test_iter)

    if gpu:
      return batch.text.cuda(), batch.label.cuda(), batch.index.cuda()

    return batch.text, batch.label, batch.index

  def idxs2sentence(self, idxs):
    return ' '.join([self.TEXT.vocab.itos[i] for i in idxs])

  def idx2label(self, idx):
    return self.LABEL.vocab.itos[idx]
