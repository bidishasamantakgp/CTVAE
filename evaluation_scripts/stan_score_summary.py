import numpy as np
import sys

scores_file = sys.argv[1]
labels_file = sys.argv[2]  # original labels

scores = np.loadtxt(scores_file)
labels = np.loadtxt(labels_file)  # original labels
# assert 2==scores.shape[1]

# scores = scores[-1000*100:]
# labels = labels[-1000*100:]

# group_len = 100

pos_label_mask = labels == 1
neg_label_mask = labels == 0
if len(set(labels)) == 3:
  pos_label_mask = labels == 2


print('num pos labels:', sum(pos_label_mask))
print('num neg labels:', sum(neg_label_mask))

# group together samples from one sentence
# upper = upper.reshape((-1, group_len))
# lower = lower.reshape((-1, group_len))

# scores are in range [0,1] with 0 for negative, 1 for positive
pos_score_mask = scores > 0.5
neg_score_mask = scores <= 0.5

# take average, count only real positive/ negative

# input_output
neg_neg_avg = np.sum(scores*neg_label_mask*neg_score_mask) / \
    np.sum(neg_label_mask*neg_score_mask)
neg_pos_avg = np.sum(scores*neg_label_mask*pos_score_mask) / \
    np.sum(neg_label_mask*pos_score_mask)
pos_neg_avg = np.sum(scores*pos_label_mask*neg_score_mask) / \
    np.sum(pos_label_mask*neg_score_mask)
pos_pos_avg = np.sum(scores*pos_label_mask*pos_score_mask) / \
    np.sum(pos_label_mask*pos_score_mask)


print()
neg_neg_ratio = np.sum(neg_label_mask*neg_score_mask) / np.sum(neg_label_mask)
neg_pos_ratio = np.sum(neg_label_mask*pos_score_mask) / np.sum(neg_label_mask)
pos_neg_ratio = np.sum(pos_label_mask*neg_score_mask) / np.sum(pos_label_mask)
pos_pos_ratio = np.sum(pos_label_mask*pos_score_mask) / np.sum(pos_label_mask)

print('input neg, output neg')
print('input neg, output pos')
print('input pos, output neg')
print('input pos, output pos')
print()
print('{:.4f} ({:.3f})'.format(1-neg_neg_avg, neg_neg_ratio))
print('{:.4f} ({:.3f})'.format(neg_pos_avg, neg_pos_ratio))
print()
print('{:.4f} ({:.3f})'.format(1-pos_neg_avg, pos_neg_ratio))
print('{:.4f} ({:.3f})'.format(pos_pos_avg, pos_pos_ratio))
# print(np.sum(upper_mask))
# print(np.sum(lower_mask))
