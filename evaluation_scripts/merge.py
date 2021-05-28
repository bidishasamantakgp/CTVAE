import sys

infiles = sys.argv[1:-1]
outfile = sys.argv[-1]


def read(fn):
  return open(fn).readlines()


in_datas = [read(fn) for fn in infiles]

with open(outfile, 'w') as f:
  for i in range(len(in_datas[0])):
    for j in range(len(in_datas)):
      f.write(in_datas[j][i])
