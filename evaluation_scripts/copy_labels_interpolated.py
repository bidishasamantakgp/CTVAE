import sys

labels_in, text_fn, reps, out_fn = sys.argv[1:1+4]

labels_ = [x.strip() for x in open(labels_in).readlines()]
text = [x.strip() for x in open(text_fn).readlines()]

labels = []
for l in labels_:
  for _ in range(int(reps)):
    labels.append(l)

with open(out_fn, 'w') as f:
  for t, l in zip(text, labels):
    if t:
      f.write(l)
      f.write('\n')
