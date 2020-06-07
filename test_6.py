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

  def filterByLabel(self):
    loi = ['new', 'unknown', 'update']  # label of interest

    train_sents_temp = []
    train_labels_temp = []
    valid_sents_temp = []
    valid_labels_temp = []
    for i, tl in enumerate(self.train_labels):
      if tl.lower() in loi:
        train_sents_temp.append(' '.join(self.train_sents[i]))
        train_labels_temp.append(self.train_labels[i])
    for i, tl in enumerate(self.valid_labels):
      if tl.lower() in loi:
        valid_sents_temp.append(' '.join(self.valid_sents[i]))
        valid_labels_temp.append(self.valid_labels[i])

    # self.train_sents, self.train_labels, self.valid_sents, self.valid_labels =
    self.train_sents = []
    self.valid_sents = []
    self.train_labels = []
    self.valid_labels = []
    self.train_sents = train_sents_temp.copy()
    self.train_labels = train_labels_temp.copy()
    self.valid_sents = valid_sents_temp.copy()
    self.valid_labels = valid_labels_temp.copy()

  def getDistinctLabel(self):
    # get distinct labels
    self.distinct_labels = list(set(self.train_labels + self.valid_labels))
    self.label_to_int = dict((l, i) for i, l in enumerate(self.distinct_labels))
    print(self.label_to_int
         )  # {'NEW': 2, 'RESOLVED': 0, 'UNKNOWN': 1, 'UPDATE': 3}

  def oneHotLabels(self):
    train_labels_int = [self.label_to_int[l] for l in self.train_labels]
    valid_labels_int = [self.label_to_int[l] for l in self.valid_labels]
    self.n_labels = len(self.distinct_labels)
    self.train_labels_onehot = []
    self.valid_labels_onehot = []
    for il in train_labels_int:
      temp = np.zeros(self.n_labels)
      temp[il] = 1
      self.train_labels_onehot.append(temp)

    for il in valid_labels_int:
      temp = np.zeros(self.n_labels)
      temp[il] = 1
      self.valid_labels_onehot.append(temp)
    print(len(self.train_labels_onehot), len(self.valid_labels_onehot))

  def randomizeData(self):
    p1 = np.random.permutation(len(self.train_labels_onehot))
    p2 = np.random.permutation(len(self.valid_labels_onehot))

    self.train_sents = np.array(self.train_sents)[p1]
    self.valid_sents = np.array(self.valid_sents)[p2]
    self.train_labels_onehot = np.array(self.train_labels_onehot)[p1]
    self.valid_labels_onehot = np.array(self.valid_labels_onehot)[p2]

    print(len(self.train_sents), len(self.valid_sents),
          len(self.train_labels_onehot), len(self.valid_labels_onehot))
    # np.save('data/train_sents_s.npy', self.train_sents)
    # np.save('data/valid_sents_s.npy', self.valid_sents)
    # np.save('data/train_labels_onehot_s.npy', self.train_labels_onehot)
    # np.save('data/valid_labels_onehot_s.npy', self.valid_labels_onehot)
    np.save('data/train_sents_xs.npy', self.train_sents)
    np.save('data/valid_sents_xs.npy', self.valid_sents)
    np.save('data/train_labels_onehot_xs.npy', self.train_labels_onehot)
    np.save('data/valid_labels_onehot_xs.npy', self.valid_labels_onehot)


if __name__ == '__main__':
  train_path = 'data/ntuc_train.json'
  valid_path = 'data/ntuc_valid.json'

  jp = JsonProcessor(train_path, valid_path)
  jp.getText()
  jp.clipLength()
  jp.processSents()
  jp.filterByLabel()
  jp.getDistinctLabel()
  jp.oneHotLabels()
  jp.randomizeData()
