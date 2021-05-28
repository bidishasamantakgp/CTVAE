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
from multiprocessing import Process, Array
from time import sleep


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


def pre(model, data, ret):
  start = datetime.now()
  _pred = model.predict(data).tolist()
  for i in range(len(_pred)):
    ret[i] = int(_pred[i])


model_path = sys.argv[1]
input_path = sys.argv[2]
output_path = sys.argv[3]
port = 8190
if len(sys.argv) > 4:
  port = int(sys.argv[4])

log('reading files')
x = read(input_path)
len_x = len(x)
print('data length:', len_x)
log('converting to BERT vectors')
bc = BertClient(ip='localhost', port=port)
xe = bc.encode(x)
log('converting to BERT vectors ...done')

log('loading model')
num_jobs = 8
models = []
for _ in range(num_jobs):
  models.append(joblib.load(model_path))

# pred = model.predict(xe)
pred = np.array(['0']*len_x)

log('starting prediction')
batch_size = 32
# Fill with dummy jobs
jobs = [Process(target=lambda x: x, args=(True,)) for _ in range(num_jobs)]
# return value, init with 9 (or any other)
# 'i' -> int
ret_vals = [Array('i', [9]*batch_size) for _ in range(num_jobs)]
starts = [0]*num_jobs
ends = [0]*num_jobs
copy = [False]*num_jobs

start = 0
end = batch_size
with tqdm(total=len(xe)) as pbar:
  while start < len(xe) or any(copy):
    for i in range(num_jobs):
      if jobs[i].is_alive():
        if i+1 == num_jobs:
          sleep(2)
        continue
      # if job has completed

      # copy answer
      if copy[i]:
        copy[i] = False
        reqd_len = len(pred[starts[i]:ends[i]])
        pred[starts[i]:ends[i]] = [str(x) for x in ret_vals[i][:reqd_len]]
        pbar.update(reqd_len)

      # run new job
      if start < len(xe):
        xe_ = xe[start:end]
        job = Process(target=pre, args=(models[i], xe_, ret_vals[i]))
        jobs[i] = job
        job.start()

        starts[i] = start
        ends[i] = end
        copy[i] = True

        start += batch_size
        end += batch_size


directory = os.path.dirname(output_path)
if directory:
  os.makedirs(directory, exist_ok=True)
with open(output_path, 'w') as f:
  for p in pred:
    f.write(p)
    f.write('\n')

with open(input_path+'.mod', 'w') as f:
  for s in x:
    f.write(s)
    f.write('\n')
