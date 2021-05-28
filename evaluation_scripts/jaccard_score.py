import sys
import numpy as np
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

original_file = sys.argv[1]
labels_file = sys.argv[2]  # original labels
output_file = sys.argv[3]  # sampled sentences
# sentiment scores [0,1] 0 for neg, 1 for pos
sentiment_scores_file = sys.argv[4]

original = [x.strip(' ?.\n') for x in open(original_file).readlines()]
labels = np.loadtxt(labels_file)  # original labels
output = [x.strip(' ?.\n') for x in open(output_file).readlines()]
sent = np.loadtxt(sentiment_scores_file)

# original = original[-1000*100:]
# labels = labels[-1000*100:]
# output = output[-1000*100:]
# sent = sent[-1000*100:]


# num_samples = 1000
num_per_sample = 100
if len(sys.argv) > 5:
  num_per_sample = int(sys.argv[5])
print('num_per_sample:', num_per_sample)

assert len(original) == len(output) == len(labels) == len(sent)
# assert len(original)==num_samples*num_per_sample

# validate the groups of num_per_sample
labels = labels.reshape((-1, num_per_sample))
# check if all labels in a row are same
assert all([len(set(group)) == 1 for group in labels])
labels = labels[:, 0]


def score(original, sampled):
  if not sampled or not original:  # blank output or blank original
    return 0

  org = set(original.split(' ')) - stop_words
  generated = set(sampled.split(' ')) - stop_words

  num = len(org.intersection(generated))
  den = len(org.union(generated))

  if den == 0:
    # print()
    # print(original, org)
    # print(sampled, generated)

    return 0

  return num/den


scores = np.array([score(original[i], output[i])
                   for i in range(len(original))])
scores = scores.reshape((-1, num_per_sample))

sent = sent.reshape((-1, num_per_sample))
sent_neg_mask = sent < 0.5
sent_pos_mask = sent > 0.5

# scores where observed sent is neg/ pos
scores_neg_sent = np.max(scores*sent_neg_mask, axis=1)
scores_pos_sent = np.max(scores*sent_pos_mask, axis=1)

labels_neg_mask = labels == 0
labels_pos_mask = labels == 1
if len(set(labels)) == 3:
  labels_pos_mask = labels == 2


# neg_neg: neg input label, and neg output label
neg_neg_scores = scores_neg_sent[labels_neg_mask]
neg_pos_scores = scores_pos_sent[labels_neg_mask]
pos_neg_scores = scores_neg_sent[labels_pos_mask]
pos_pos_scores = scores_pos_sent[labels_pos_mask]

print('negative input, negative output')
print('negative input, positive output')
print('positive input, negative output')
print('positive input, positive output')
print()

print('{:.3f}, {:.4f}'.format(np.mean(neg_neg_scores), np.std(neg_neg_scores)))
print('{:.3f}, {:.4f}'.format(np.mean(neg_pos_scores), np.std(neg_pos_scores)))
print()
print('{:.3f}, {:.4f}'.format(np.mean(pos_neg_scores), np.std(pos_neg_scores)))
print('{:.3f}, {:.4f}'.format(np.mean(pos_pos_scores), np.std(pos_pos_scores)))
