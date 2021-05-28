# bert-serving-start -model_dir bert-base-uncased -num_worker=5 -port=8190 -max_seq_len=NONE

from bert_serving.client import BertClient
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
import joblib
import sys
import numpy as np


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


model_path = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]

log('reading files')
x = read(input_path)
print('data length:', len(x))
log('converting to BERT vectors')
bc = BertClient(ip='localhost', port=8190)
xe = bc.encode(x)

log('loading model:', model_path)
model = joblib.load(model_path)
log('calling .predict()')
probs = model.predict_proba(xe)

directory = os.path.dirname(output_path)
if directory:
  os.makedirs(directory, exist_ok=True)

np.savetxt(output_path, probs)

# with open(output_path, 'w') as f:
#  for p in pred:
#    f.write(p)
#    f.write('\n')


exit()
################
with open(input_path+'.mod', 'w') as f:
  for s in x:
    f.write(s)
    f.write('\n')
