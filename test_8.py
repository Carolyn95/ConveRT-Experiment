# test MixProcessor
import numpy as np
import pdb


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
