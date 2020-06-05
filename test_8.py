# test MixProcessor
import numpy as np


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
    np.save(self.train_data, 'data/train_sents_mixed.npy')
    np.save(self.train_label, 'data/train_labels_onehot_mixed.npy')
    np.save(self.valid_data, 'data/valid_sents_mixed.npy')
    np.save(self.valid_label, 'data/valid_labels_onehot_mixed.npy')


if __name__ == '__main__':
  train_paths = [
      'data/train_sents_l.npy', 'data/train_sents_s.npy',
      'data/train_labels_onehot_l.npy', 'data/train_labels_onehot_s.npy'
  ]
  valid_paths = [
      'data/valid_sents_l.npy', 'data/valid_sents_s.npy',
      'data/valid_labels_onehot_l.npy', 'data/valid_labels_onehot_s.npy'
  ]
  mp = MixProcessor(train_paths, valid_paths)
  mp.saveData()
