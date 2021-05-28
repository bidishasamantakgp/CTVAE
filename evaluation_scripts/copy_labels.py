import sys

labels_in, reps, out_fn = sys.argv[1:1+3]

labels_ = [x.strip() for x in open(labels_in).readlines()]

labels = []
for l in labels_:
  for _ in range(int(reps)):
    labels.append(l)

with open(out_fn, 'w') as f:
  for l in labels:
    f.write(l)
    f.write('\n')
