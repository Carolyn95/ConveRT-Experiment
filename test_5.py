"""
# this is for examining sentences with number of tokens > 40
# are they meaningful?

# sentences in every email (one email may contain more than one sentence)
"""

import gluonnlp as nlp
import multiprocessing as mp
import pdb
from rich import print
import time
from mxnet import gluon
import numpy as np

MAX_TOKENS = 40


# process large
class NpyProcessor():
  # should include a split train/valid method

  def __init__(self, data_path):
    self.tokenizer = nlp.data.SpacyTokenizer('en')
    self.length_clip = nlp.data.ClipSequence(MAX_TOKENS)
    self.data = np.load(data_path, allow_pickle=True)


if __name__ == '__main__':

  data_path = 'data/NoLongEmail_email.npy'

  np = NpyProcessor(data_path)
  # email_counter = 1
  counter = 1
  for idx, data in enumerate(np.data):
    for sent in data:
      sent_len = len(np.tokenizer(sent))
      if sent_len > MAX_TOKENS:
        counter += 1
        print(idx, ':', sent_len, '=>', sent)
  print(counter)
""" # result

"""