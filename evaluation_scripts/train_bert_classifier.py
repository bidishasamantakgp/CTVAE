# bert-serving-start -model_dir bert-base-uncased -num_worker=5 -port=8190 -max_seq_len=NONE

from bert_serving.client import BertClient
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import os
from datetime import datetime
import joblib
import sys
from tqdm import tqdm
import numpy as np


def read(fp):
  with open(fp) as f:
    return [x.strip() for x in f.readlines()]


def log(*args):
  print(datetime.now(), *args)


train_dir = sys.argv[1]
val_dir = sys.argv[2]
model_output_file = sys.argv[3]

port = 8190
if len(sys.argv) > 4:
  port = int(sys.argv[4])

data_fn = 'data.txt'
labels_fn = 'labels.txt'


log('reading files')
x = read(os.path.join(train_dir, data_fn))
y = read(os.path.join(train_dir, labels_fn))

x_val = read(os.path.join(val_dir, data_fn))
y_val = read(os.path.join(val_dir, labels_fn))


log('converting to BERT vectors')
bc = BertClient(ip='localhost', port=port)
# xe = bc.encode(x)
xe = np.zeros((len(x), 768))

batch_size = 128*64

log('getting BERT encoding for x')
log('length x', len(x))
start = 0
end = batch_size
with tqdm(total=len(x)) as pbar:
  while start < len(x):
    x_ = x[start:end]
    xe_ = bc.encode(x_)
    xe[start:end] = xe_

    pbar.update(len(xe[start:end]))
    start += batch_size
    end += batch_size
log('length xe', len(xe))
log('getting BERT encoding for x_val')
xe_val = bc.encode(x_val)
log('getting BERT encoding for x_val ..done')


model = SVC()
log('Starting training.')
model.fit(xe, y)
log('Getting val accuracy.')
pred = model.predict(xe_val)
log('Accuracy:', accuracy_score(y_val, pred))

os.makedirs(os.path.dirname(model_output_file), exist_ok=True)
joblib.dump(model, model_output_file)
