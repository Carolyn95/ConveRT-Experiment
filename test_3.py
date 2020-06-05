# Data Processor (JsonProcessor | ArrayProcessor | MixProcessor)
import gluonnlp as nlp
import multiprocessing as mp
import pdb
from rich import print
import time
from mxnet import gluon
import numpy as np

MAX_TOKENS = 40


# process small
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
          print(train_sent[idx])

    for valid_sent in self.valid_sents:
      for idx, sent in enumerate(valid_sent):
        tokenized_sent = self.tokenizer(sent)
        if len(tokenized_sent) > 40:
          tokenized_sent = self.length_clip(tokenized_sent)
          valid_sent[idx] = ' '.join(tokenized_sent)
          print(valid_sent[idx])

  def processSents(self):
    for idx, t in enumerate(self.train_sents):
      content = '. '.join(t)
      self.train_sents[idx] = content

    for idx, t in enumerate(self.valid_sents):
      content = '. '.join(t)
      self.valid_sents[idx] = content

  def getDistinctLabel(self):
    # get distinct labels
    self.distinct_labels = list(set(self.train_labels + self.valid_labels))
    self.label_to_int = dict((l, i) for i, l in enumerate(self.distinct_labels))
    print(self.label_to_int)

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
    np.save('data/train_sents_s.npy', self.train_sents)
    np.save('data/valid_sents_s.npy', self.valid_sents)
    np.save('data/train_labels_onehot_s.npy', self.train_labels_onehot)
    np.save('data/valid_labels_onehot_s.npy', self.valid_labels_onehot)


# process large
class ArrayProcessor():
  # should include a split train/valid method

  def __init__(self, sents_path, labels_path):
    self.tokenizer = nlp.data.SpacyTokenizer('en')
    self.length_clip = nlp.data.ClipSequence(MAX_TOKENS)
    self.data = np.load(sents_path, allow_pickle=True)
    self.labels = np.load(labels_path, allow_pickle=True)

  def clipLength(self):
    # Clip the length of email words

    for data in self.data:
      for idx, sent in enumerate(data):
        tokenized_sent = self.tokenizer(sent)
        if len(tokenized_sent) > 40:
          tokenized_sent = self.length_clip(tokenized_sent)
          data[idx] = ' '.join(tokenized_sent)
          print(idx, data[idx])

  def processSents(self):
    for idx, t in enumerate(self.data):
      content = '. '.join(t)
      self.data[idx] = content

  def getDistinctLabel(self):
    # get distinct labels
    self.distinct_labels = list(set(self.labels))
    self.label_to_int = dict((l, i) for i, l in enumerate(self.distinct_labels))
    print(self.label_to_int)  # {'new': 2, 'unknown': 1, 'update': 0}

  def oneHotLabels(self):
    labels_int = [self.label_to_int[l] for l in self.labels]
    self.n_labels = len(self.distinct_labels)
    self.labels_onehot = []
    for il in labels_int:
      temp = np.zeros(self.n_labels)
      temp[il] = 1
      self.labels_onehot.append(temp)

  def randomizeData(self):
    p = np.random.permutation(len(self.labels_onehot))
    train_split = int(len(p) * 0.8)

    self.sents = np.array(self.data)[p]
    self.labels_onehot = np.array(self.labels_onehot)[p]
    self.train_sents = self.sents[:train_split]
    self.valid_sents = self.sents[train_split:]
    self.train_labels_onehot = self.labels_onehot[:train_split]
    self.valid_labels_onehot = self.labels_onehot[train_split:]

    print(len(self.train_sents), len(self.valid_sents),
          len(self.train_labels_onehot),
          len(self.valid_labels_onehot))  # 5324 1331 5324 1331
    np.save('data/train_sents_l.npy', self.train_sents)
    np.save('data/valid_sents_l.npy', self.valid_sents)
    np.save('data/train_labels_onehot_l.npy', self.train_labels_onehot)
    np.save('data/valid_labels_onehot_l.npy', self.valid_labels_onehot)


# combine the two
class MixProcessor():

  def __init__(self, train_paths, valid_paths):
    self.train_data = np.concatenate(
        [np.load(f, allow_pickle=True) for f in train_paths if 'sents' in f])
    self.train_label = np.concatenate(
        [np.load(f, allow_pickle=True) for f in train_paths if 'labels' in f])
    self.valid_data = np.concatenate(
        [np.load(f, allow_pickle=True) for f in valid_paths if 'sents' in f])
    self.valid_label = np.concatenate(
        [np.load(f, allow_pickle=True) for f in valid_paths if 'labels' in f])
    print()

  def saveData(self):
    np.save('data/train_sents_mixed.npy', self.train_data)
    np.save('data/train_labels_onehot_mixed.npy', self.train_label)
    np.save('data/valid_sents_mixed.npy', self.valid_data)
    np.save('data/valid_labels_onehot_mixed.npy', self.valid_label)


if __name__ == '__main__':
  train_path = 'data/ntuc_train.json'
  valid_path = 'data/ntuc_valid.json'

  jp = JsonProcessor(train_path, valid_path)
  jp.getText()
  jp.clipLength()
  jp.processSents()
  jp.getDistinctLabel()
  jp.oneHotLabels()
  jp.randomizeData()

  sents_path = 'data/NoLongEmail_email.npy'
  labels_path = 'data/NoLongEmail_label.npy'
  ap = ArrayProcessor(sents_path, labels_path)
  ap.clipLength()
  ap.processSents()
  ap.getDistinctLabel()
  ap.oneHotLabels()
  ap.randomizeData()

  train_paths = [
      'data/train_sents_l.npy', 'data/train_sents_xs.npy',
      'data/train_labels_onehot_l.npy', 'data/train_labels_onehot_xs.npy'
  ]
  valid_paths = [
      'data/valid_sents_l.npy', 'data/valid_sents_xs.npy',
      'data/valid_labels_onehot_l.npy', 'data/valid_labels_onehot_xs.npy'
  ]
  mp = MixProcessor(train_paths, valid_paths)
  mp.saveData()
