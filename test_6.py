# testing JsonProcessor
import gluonnlp as nlp
import multiprocessing as mp
import pdb
from rich import print
import time
from mxnet import gluon
import numpy as np
MAX_TOKENS = 40


class JsonProcessor():

  def __init__(self, train_path, valid_path):
    self.tokenizer = nlp.data.SpacyTokenizer('en')
    self.length_clip = nlp.data.ClipSequence(MAX_TOKENS)
    with open(train_path, 'r') as f:
      train = f.read()
      self.train = train.splitlines()
    with open(valid_path, 'r') as f:
      valid = f.read()
      self.valid = valid.splitlines()

  def getText(self):
    # get data into format => ['string', 'string']
    self.train_sents = []
    self.valid_sents = []
    self.train_labels = []
    self.valid_labels = []

    for t in self.train:
      t = eval(t)
      content = t['content']
      if len(content) != 0:
        # content = '.'.join(content)
        self.train_sents.append(content)
        self.train_labels.append(t['intent'])

    for v in self.valid:
      v = eval(v)
      content = v['content']
      if len(content) != 0:
        # content = '.'.join(content)
        self.valid_sents.append(content)
        self.valid_labels.append(v['intent'])

  def clipLength(self):
    # Clip the length of email words
    for train_sent in self.train_sents:
      for idx, sent in enumerate(train_sent):
        tokenized_sent = self.tokenizer(sent)
        if len(tokenized_sent) > 40:
          tokenized_sent = self.length_clip(tokenized_sent)
          train_sent[idx] = ' '.join(tokenized_sent)
          pdb.set_trace()
          print(train_sent[idx])

  def processSents(self):
    for idx, t in enumerate(self.train_sents):
      content = '. '.join(t)
      self.train_sents[idx] = content


if __name__ == '__main__':
  train_path = 'data/ntuc_train.json'
  valid_path = 'data/ntuc_valid.json'

  jp = JsonProcessor(train_path, valid_path)
  jp.getText()
  jp.clipLength()
  jp.processSents()
  pdb.set_trace()