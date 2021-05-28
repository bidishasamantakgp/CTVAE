import glob
import io
import os

from torchtext import data


class Family(data.Dataset):

    #name = 'yelp_new'
    #dirname = 'yelp_new'
    name = 'family'
    dirname = 'family'
    #name = 'amazon_tense'
    #dirname = 'amazon_tense'
    @staticmethod
    def sort_key(ex):
        return len(ex.text)

    def __init__(self, path, text_field, label_field, index_field, **kwargs):
        """Create an yelp dataset instance given a path and fields.
        Arguments:
            path: Path to the dataset's highest level directory
            text_field: The field that will be used for text data.
            label_field: The field that will be used for label data.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        #path = './data/'+dirname+"/"+name+"/"
        fields = [('text', text_field), ('label', label_field), ('index', index_field)]
        examples = []

        sentences = open(os.path.join(path, 'data.txt'))
        labels = open(os.path.join(path, 'labels.txt'))
        # for label in ['pos', 'neg']:
        #     for fname in glob.iglob(os.path.join(path, label, '*.txt')):
        #         with io.open(fname, 'r', encoding="utf-8") as f:
        #             text = f.readline()
        #         examples.append(data.Example.fromlist([text, label], fields))
        count = 0
        for _sent, _label in zip(sentences, labels):
            examples.append(data.Example.fromlist([_sent, int(_label), count], fields))
            count+=1

        super(Family, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, text_field, label_field, index_field, root='.data',
               train='train', val='val', test='test', **kwargs):
        """Create dataset objects for splits of the IMDB dataset.
        Arguments:
            text_field: The field that will be used for the sentence.
            label_field: The field that will be used for label data.
            root: Root dataset storage directory. Default is '.data'.
            train: The directory that contains the training examples
            test: The directory that contains the test examples
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        return super(Family, cls).splits(
            root=root, text_field=text_field, label_field=label_field, index_field=index_field,
            train=train, validation=val, test=test, **kwargs)


    @classmethod
    def iters(cls, batch_size=32, device=0, root='.data', vectors=None, **kwargs):
        """Create iterator objects for splits of the Yelp dataset.
        Arguments:
            batch_size: Batch_size
            device: Device to create batches on. Use - 1 for CPU and None for
                the currently active GPU device.
            root: The root directory that contains the yelp dataset subdirectory
            vectors: one of the available pretrained vectors or a list with each
                element one of the available pretrained vectors (see Vocab.load_vectors)
            Remaining keyword arguments: Passed to the splits method.
        """
        TEXT = data.Field()
        LABEL = data.Field(sequential=False)
        INDEX = data.Field(sequential=False)

        train, test = cls.splits(TEXT, LABEL, INDEX, root=root, **kwargs)

        TEXT.build_vocab(train, vectors=vectors)
        LABEL.build_vocab(train)
        INDEX.build_vocab(train)
        return data.BucketIterator.splits(
            (train, test), batch_size=batch_size, device=device)
