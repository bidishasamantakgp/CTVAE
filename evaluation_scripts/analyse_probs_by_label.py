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

print('num 0 labels:', sum(neg_label_mask))
print('num 1 labels:', sum(pos_label_mask))
print('total labels:', sum(pos_label_mask)+sum(neg_label_mask))

# group together samples from one sentence
# upper = upper.reshape((-1, group_len))
# lower = lower.reshape((-1, group_len))

# scores are in [score for 0, score for 1], with sum 1
pos_score_mask = scores[:, 1] > 0.5
neg_score_mask = scores[:, 0] >= 0.5

# take average, count only real positive/ negative

# input_output
neg_neg_avg = np.sum(scores[:, 0]*neg_label_mask *
                     neg_score_mask)/np.sum(neg_label_mask*neg_score_mask)
neg_pos_avg = np.sum(scores[:, 1]*neg_label_mask *
                     pos_score_mask)/np.sum(neg_label_mask*pos_score_mask)
pos_neg_avg = np.sum(scores[:, 0]*pos_label_mask *
                     neg_score_mask)/np.sum(pos_label_mask*neg_score_mask)
pos_pos_avg = np.sum(scores[:, 1]*pos_label_mask *
                     pos_score_mask)/np.sum(pos_label_mask*pos_score_mask)


print()
neg_neg_ratio = np.sum(neg_label_mask*neg_score_mask) / np.sum(neg_label_mask)
neg_pos_ratio = np.sum(neg_label_mask*pos_score_mask) / np.sum(neg_label_mask)
pos_neg_ratio = np.sum(pos_label_mask*neg_score_mask) / np.sum(pos_label_mask)
pos_pos_ratio = np.sum(pos_label_mask*pos_score_mask) / np.sum(pos_label_mask)


neg_avg = np.sum(scores[:, 0]*neg_score_mask)/np.sum(neg_score_mask)
pos_avg = np.sum(scores[:, 1]*pos_score_mask)/np.sum(pos_score_mask)

neg_ratio = np.mean(neg_score_mask)
pos_ratio = np.mean(pos_score_mask)


print('output 0')
print('output 1')
print()
print('input 0, output 0')
print('input 0, output 1')
print()
print('input 1, output 0')
print('input 1, output 1')
print()
print('{:.4f} ({:.3f})'.format(neg_avg, neg_ratio))
print('{:.4f} ({:.3f})'.format(pos_avg, pos_ratio))
print()
print('{:.4f} ({:.3f})'.format(neg_neg_avg, neg_neg_ratio))
print('{:.4f} ({:.3f})'.format(neg_pos_avg, neg_pos_ratio))
print()
print('{:.4f} ({:.3f})'.format(pos_neg_avg, pos_neg_ratio))
print('{:.4f} ({:.3f})'.format(pos_pos_avg, pos_pos_ratio))
# print(np.sum(upper_mask))
# print(np.sum(lower_mask))
