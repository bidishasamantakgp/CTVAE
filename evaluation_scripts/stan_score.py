import sys
from pycorenlp import StanfordCoreNLP
from tqdm import tqdm
# pip install pycorenlp tqdm

# DOWNLOAD
# wget http://nlp.stanford.edu/software/stanford-corenlp-full-2018-10-05.zip
# unzip stanford-corenlp-full-2018-10-05.zip
# wget http://nlp.stanford.edu/software/stanford-english-corenlp-2018-10-05-models.jar
# RUN
# java -mx6g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 5000

port = '9100'

in_file, out_sent = sys.argv[1:1 + 2]
if len(sys.argv) > 3:
      port, in_file, out_sent = sys.argv[1:1 + 3]
      # in_file, input sentences
      # out_sent, sentiment values between 0,1

nlp = StanfordCoreNLP('http://127.0.0.1:'+port)


def analyze(text):
  return nlp.annotate(
      text,
      properties={
          'annotators': 'sentiment',
          'outputFormat': 'json',
          'timeout': 5000,
      }
  )


num_lines = len(open(in_file).readlines())

with open(in_file) as f, open(out_sent, 'w') as sf:
  for line in tqdm(f, total=num_lines):
    line = line.strip()

    sent_score=0
    if line:
      result = analyze(line)
      # for s in result['sentences']:
      try:
        s = result['sentences']
        if s:
          s = s[0]
          # sentence = ' '.join([t['word'] for t in s['tokens']])
          # sent_value = s['sentimentValue']
          # sentiment = s['sentiment']
          sent_score = sum([i*x for i, x in enumerate(s['sentimentDistribution'])])/4
      except:
        pass

    sf.write(str(sent_score))
    sf.write('\n')
