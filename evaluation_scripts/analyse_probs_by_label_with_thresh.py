import numpy as np
import sys
from bert_serving.client import BertClient
from datetime import datetime
from itertools import groupby
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

clf_scores_file = sys.argv[1]
orig_labels_file = sys.argv[2]  # original labels

original_sentences_file = sys.argv[3]
generated_sentences_file = sys.argv[4]

# THRESHOLD
cosine_th = float(sys.argv[5])
overlap_th = float(sys.argv[6])


# cosine_out_file = sys.argv[5]

# NOTE: auto detection of groups based on original sentences

# group_len = 10
# group_len = sys.argv[5]

clf_scores = np.loadtxt(clf_scores_file)
orig_labels = np.loadtxt(orig_labels_file)  # original labels
assert 2 == clf_scores.shape[1]


def read(fp):
  s = []
  with open(fp) as f:
    for x in f.readlines():
      x = x.strip()
      if x:
        s.append(x)
  return s


def log(*args):
  print(datetime.now(), *args)


orig_sentences = read(original_sentences_file)
gen_sentences = read(generated_sentences_file)

# create duplicate for backup/ later use
orig_sentences_ = orig_sentences[:]
orig_labels_ = np.copy(orig_labels)
clf_scores_ = np.copy(clf_scores)

print(len(orig_sentences))
print(len(gen_sentences))

assert len(orig_sentences) == len(gen_sentences)
assert len(orig_sentences) == len(orig_labels)
assert len(orig_sentences) == len(clf_scores)

# get BERT vectors
log('converting to BERT vectors')
bc = BertClient(ip='localhost', port=8190)
orig_bert = bc.encode(orig_sentences)
gen_bert = bc.encode(gen_sentences)
log('converting to BERT vectors ...done')

# get cosine with BERT vectors


def calc_cosine(a, b):
  return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))


def get_cosine_scores(a_list, b_list):
  assert len(a_list) == len(b_list)
  scores = [0]*len(a_list)
  for i in range(len(a_list)):
    scores[i] = calc_cosine(a_list[i], b_list[i])
  return scores


print()
log('calculating cosine scores')
cosine_scores = get_cosine_scores(orig_bert, gen_bert)
log('calculating cosine scores ...done')

# with open(cosine_out_file, 'w') as f:
#   for x in cosine_scores:
#     f.write(str(x))
#     f.write('\n')
# exit()


# get overlap scores

def score(original, sampled):
  if not sampled or not original:  # blank output or blank original
    return 0
  org = original.split(' ')
  generated = sampled.split(' ')
  generated = [x for x in generated if x not in stop_words]
  if not generated:
    return 0
  return len([x for x in generated if x in org])/len(generated)


overlap_scores = [score(orig_sentences[i], gen_sentences[i])
                  for i in range(len(orig_sentences))]

assert len(orig_sentences) == len(overlap_scores)
assert len(orig_sentences) == len(cosine_scores)


indices = [cosine_scores[i] > cosine_th and overlap_scores[i] > overlap_th
           for i in range(len(orig_sentences))]


def filter_by_indices(a, idxs):
  return [a[i] for i in range(len(idxs)) if idxs[i]]


print()
orig_sentences = filter_by_indices(orig_sentences, indices)
gen_sentences = filter_by_indices(gen_sentences, indices)
orig_labels = orig_labels[indices]
clf_scores = clf_scores[indices]
cosine_scores = filter_by_indices(cosine_scores, indices)
overlap_scores = filter_by_indices(overlap_scores, indices)

assert len(orig_sentences) == len(orig_labels)

# GROUPING

# in case original sentence is blank,
# we may group together multiple consecutive groups
# hence, prepend with label to try to make them unique
group_lengths = []
labels_ = [str(x) for x in orig_labels.tolist()]
for_groupby = [labels_[i]+orig_sentences[i]
               for i in range(len(orig_sentences))]
for _, grp in groupby(for_groupby):
  group_lengths.append(len(list(grp)))

print('\nno. of groups:', len(group_lengths))


def collect(arr, lengths):
  res = []
  i = 0
  for l in lengths:
    new_i = i+l
    res.append(np.array(arr[i:new_i]))
    i = new_i
  return np.array(res)


# verify same orig label for a group
orig_labels = collect(orig_labels, group_lengths)
for grp in orig_labels.tolist():
  if not isinstance(grp, list):
    grp = grp.tolist()
  assert len(set(grp)) == 1
orig_labels = np.array([x[0] for x in orig_labels.tolist()])
assert len(orig_labels) == len(group_lengths)

# single value per group
orig_0_mask = orig_labels == 0
orig_1_mask = orig_labels == 1

# value per elem of group
# for clf scores mask
clf_0_mask = collect(clf_scores[:, 0] >= 0.5, group_lengths)
clf_1_mask = collect(clf_scores[:, 1] > 0.5, group_lengths)
# also do clf_scores
clf_0_scores = collect(clf_scores[:, 0], group_lengths)
clf_1_scores = collect(clf_scores[:, 1], group_lengths)
# above was clf score for all,
# now apply mask, so that we get 0 score only for that where it is 0
# NOTE: following is a lists of lists (np array)
# NOTE: and can contain empty lists
clf_0_scores = np.array([clf_0_scores[i][clf_0_mask[i]]
                         for i in range(len(clf_0_scores))])
clf_1_scores = np.array([clf_1_scores[i][clf_1_mask[i]]
                         for i in range(len(clf_1_scores))])


# for cosine and overlap scores
cosine_scores = collect(cosine_scores, group_lengths)
overlap_scores = collect(overlap_scores, group_lengths)


def get_max(nparr):
  # get max for each sub array
  # NOTE: if a sub array is empty, ignore it
  arr = nparr.tolist()
  res = np.array([np.max(x) for x in arr if len(x)])
  return res


orig_all_clf_0_scores = get_max(clf_0_scores)
orig_all_clf_1_scores = get_max(clf_1_scores)

orig_0_clf_0_scores = get_max(clf_0_scores[orig_0_mask])
orig_0_clf_1_scores = get_max(clf_1_scores[orig_0_mask])
orig_1_clf_0_scores = get_max(clf_0_scores[orig_1_mask])
orig_1_clf_1_scores = get_max(clf_1_scores[orig_1_mask])


# NOTE: the denominator for ratio calculation should be the number
# before any kind of filteration
# orig_labels, orig_0_mask, orig_1_mask are all AFTER filteration
# So, do NOT use them for denominator

# CALC for ratio denominator
group_lengths_ = []
_labels_ = [str(x) for x in orig_labels_.tolist()]
for_groupby_ = [_labels_[i]+orig_sentences_[i]
                for i in range(len(orig_sentences_))]
for _, grp in groupby(for_groupby_):
  group_lengths_.append(len(list(grp)))

orig_labels_ = collect(orig_labels_, group_lengths_)
for grp in orig_labels_.tolist():
  if not isinstance(grp, list):
    grp = grp.tolist()
  assert len(set(grp)) == 1
orig_labels_ = np.array([x[0] for x in orig_labels_.tolist()])
assert len(orig_labels_) == len(group_lengths_)

# single value per group
orig_0_mask_ = orig_labels_ == 0
orig_1_mask_ = orig_labels_ == 1
######

# ratio of 0 out wrt all input
orig_all_clf_0_ratio = len(orig_all_clf_0_scores)/len(orig_labels_)
# ratio of 1 out wrt all input
orig_all_clf_1_ratio = len(orig_all_clf_1_scores)/len(orig_labels_)

# ratio of 0 out wrt all 0 input
orig_0_clf_0_ratio = len(orig_0_clf_0_scores)/np.sum(orig_0_mask_)
# ratio of 1 out wrt all 0 input
orig_0_clf_1_ratio = len(orig_0_clf_1_scores)/np.sum(orig_0_mask_)
# ratio of 0 out wrt all 1 input
orig_1_clf_0_ratio = len(orig_1_clf_0_scores)/np.sum(orig_1_mask_)
# ratio of 1 out wrt all 1 input
orig_1_clf_1_ratio = len(orig_1_clf_1_scores)/np.sum(orig_1_mask_)

print('output 0')
print('output 1')
print()
print('input 0, output 0')
print('input 0, output 1')
print()
print('input 1, output 0')
print('input 1, output 1')
print()

print('{:.4f} ({:.3f})'.format(np.mean(orig_all_clf_0_scores),
                               orig_all_clf_0_ratio))
print('{:.4f} ({:.3f})'.format(np.mean(orig_all_clf_1_scores),
                               orig_all_clf_1_ratio))
print()
print('{:.4f} ({:.3f})'.format(np.mean(orig_0_clf_0_scores),
                               orig_0_clf_0_ratio))
print('{:.4f} ({:.3f})'.format(np.mean(orig_0_clf_1_scores),
                               orig_0_clf_1_ratio))
print()
print('{:.4f} ({:.3f})'.format(np.mean(orig_1_clf_0_scores),
                               orig_1_clf_0_ratio))
print('{:.4f} ({:.3f})'.format(np.mean(orig_1_clf_1_scores),
                               orig_1_clf_1_ratio))

# print('\nlen clf_0_scores:', len(clf_0_scores))
# print('len clf_1_scores:', len(clf_1_scores))


orig_sentences = collect(orig_sentences, group_lengths)
for grp in orig_sentences.tolist():
  if not isinstance(grp, list):
    grp = grp.tolist()
  assert len(set(grp)) == 1
orig_sentences = np.array([x[0] for x in orig_sentences.tolist()])
assert len(orig_sentences) == len(group_lengths)

with open('cosine/'+generated_sentences_file+'.original_sentences', 'w') as f:
  for s in orig_sentences:
    f.write(s)
    f.write('\n')

with open('cosine/'+generated_sentences_file+'.original_labels', 'w') as f:
  for s in orig_labels:
    f.write(str(s))
    f.write('\n')

gen_sentences = collect(gen_sentences, group_lengths)
clf_0_gen_sentences = np.array([gen_sentences[i][clf_0_mask[i]]
                                for i in range(len(gen_sentences))])
clf_1_gen_sentences = np.array([gen_sentences[i][clf_1_mask[i]]
                                for i in range(len(gen_sentences))])

clf_0_cosine_scores = np.array([cosine_scores[i][clf_0_mask[i]]
                                for i in range(len(cosine_scores))])
clf_1_cosine_scores = np.array([cosine_scores[i][clf_1_mask[i]]
                                for i in range(len(cosine_scores))])

clf_0_overlap_scores = np.array([overlap_scores[i][clf_0_mask[i]]
                                 for i in range(len(overlap_scores))])
clf_1_overlap_scores = np.array([overlap_scores[i][clf_1_mask[i]]
                                 for i in range(len(overlap_scores))])


def get_max_idx(nparr):
  arr = nparr.tolist()
  res = np.array([np.argmax(x) if len(x) else -1 for x in arr])
  return res


# print('len(orig_sentences):', len(orig_sentences))
# print('len(orig_labels):', len(orig_labels))

clf_0_indices = get_max_idx(clf_0_scores)
clf_1_indices = get_max_idx(clf_1_scores)

# print('len(clf_0_indices):', len(clf_0_indices))

# print('len(clf_0_gen_sentences):', len(clf_0_gen_sentences))
# print('len(clf_0_cosine_scores):', len(clf_0_cosine_scores))
# print('len(clf_0_overlap_scores):', len(clf_0_overlap_scores))

# print('\nlen(clf_1_indices):', len(clf_1_indices))
# print('len(clf_1_gen_sentences):', len(clf_1_gen_sentences))
# print('len(clf_1_cosine_scores):', len(clf_1_cosine_scores))
# print('len(clf_1_overlap_scores):', len(clf_1_overlap_scores))

# exit()

with open('cosine/'+generated_sentences_file+'.selected_0', 'w') as f:
  for i in range(len(clf_0_indices)):
    idx = clf_0_indices[i]
    if idx != -1:
      f.write(clf_0_gen_sentences[i][idx])
    f.write('\n')

with open('cosine/'+generated_sentences_file+'.selected_0_cosine', 'w') as f:
  for i in range(len(clf_0_indices)):
    idx = clf_0_indices[i]
    if idx != -1:
      f.write(str(clf_0_cosine_scores[i][idx]))
    f.write('\n')

with open('cosine/'+generated_sentences_file+'.selected_0_overlap', 'w') as f:
  for i in range(len(clf_0_indices)):
    idx = clf_0_indices[i]
    if idx != -1:
      f.write(str(clf_0_overlap_scores[i][idx]))
    f.write('\n')

with open('cosine/'+generated_sentences_file+'.selected_0_clf', 'w') as f:
  for i in range(len(clf_0_indices)):
    idx = clf_0_indices[i]
    if idx != -1:
      f.write(str(clf_0_scores[i][idx]))
    f.write('\n')


with open('cosine/'+generated_sentences_file+'.selected_1', 'w') as f:
  for i in range(len(clf_1_indices)):
    idx = clf_1_indices[i]
    if idx != -1:
      f.write(clf_1_gen_sentences[i][idx])
    f.write('\n')

with open('cosine/'+generated_sentences_file+'.selected_1_cosine', 'w') as f:
  for i in range(len(clf_1_indices)):
    idx = clf_1_indices[i]
    if idx != -1:
      f.write(str(clf_1_cosine_scores[i][idx]))
    f.write('\n')

with open('cosine/'+generated_sentences_file+'.selected_1_overlap', 'w') as f:
  for i in range(len(clf_1_indices)):
    idx = clf_1_indices[i]
    if idx != -1:
      f.write(str(clf_1_overlap_scores[i][idx]))
    f.write('\n')

with open('cosine/'+generated_sentences_file+'.selected_1_clf', 'w') as f:
  for i in range(len(clf_1_indices)):
    idx = clf_1_indices[i]
    if idx != -1:
      f.write(str(clf_1_scores[i][idx]))
    f.write('\n')
