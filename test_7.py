# testing NpyProcessor
import gluonnlp as nlp
import multiprocessing as mp
import pdb
from rich import print
import time
from mxnet import gluon
import numpy as np
MAX_TOKENS = 40


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


if __name__ == '__main__':
  sents_path = 'data/NoLongEmail_email.npy'
  labels_path = 'data/NoLongEmail_label.npy'
  ap = ArrayProcessor(sents_path, labels_path)
  ap.clipLength()
  pdb.set_trace()
  ap.processSents()
  ap.getDistinctLabel()
  ap.oneHotLabels()
  ap.randomizeData()
