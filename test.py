import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove logging info
import warnings
warnings.filterwarnings('ignore')  # filter out warnings

import encoder_client
import pdb
import numpy as np
import tensorflow as tf
import keras.backend as K
import tensorflow_hub as hub
from keras.models import Model
from keras.regularizers import l1
from keras.layers import Input, Lambda, Dense, Dropout, Reshape, Flatten
from keras.callbacks import ModelCheckpoint, TensorBoard
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import random


class DataProcessor():

  def __init__(self, train_path, valid_path):
    with open(train_path, 'r') as f:
      train = f.read()
      self.train = train.splitlines()
    with open(valid_path, 'r') as f:
      valid = f.read()
      self.valid = valid.splitlines()

  def processText(self):
    # ConveRT requires input data ['string', 'string']
    self.train_sents = []
    self.valid_sents = []
    self.train_labels = []
    self.valid_labels = []

    for t in self.train:
      t = eval(t)
      content = t['content']
      if len(content) != 0:
        content = '.'.join(content)
        self.train_sents.append(content)
        self.train_labels.append(t['intent'])

    for v in self.valid:
      v = eval(v)
      content = v['content']
      if len(content) != 0:
        content = '.'.join(content)
        self.valid_sents.append(content)
        self.valid_labels.append(v['intent'])

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


class VanillaConveRT():

  def __init__(self, train_x, train_y, valid_x, valid_y):
    self.train_x = train_x
    self.train_y = train_y
    self.valid_x = valid_x
    self.valid_y = valid_y
    self.n_labels = self.valid_y.shape[1]
    self.module = hub.Module(
        "http://models.poly-ai.com/convert/v1/model.tar.gz")
    # self.module = encoder_client.EncoderClient("http://models.poly-ai.com/convert/v1/model.tar.gz")

  def encode(self, x):
    return self.module(tf.reshape(tf.cast(x, 'string'), [-1]),
                       signature='encode_sequence',
                       as_dict=True)['sequence_encoding']

  def createModel(self):
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(self.encode, output_shape=(512,))(input_text)
    dense = Dense(256, activation='relu',
                  kernel_regularizer=l1(0.0001))(embedding)
    dense = Dropout(0.4)(dense)
    pred = Dense(self.n_labels, activation='softmax')(dense)
    self.model = Model(inputs=input_text, outputs=pred)
    self.model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    print(self.model.summary())

  def train(self, filepath):
    try:
      os.mkdir(filepath)
    except:
      pass

    with tf.Session() as sess:
      K.set_session(sess)
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      ckpt = ModelCheckpoint(filepath + '/{epoch:02d}.hdf5',
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True,
                             mode='auto')
      hist = self.model.fit(self.train_x,
                            self.train_y,
                            validation_split=0.2,
                            epochs=20,
                            batch_size=32,
                            callbacks=[ckpt])

      pred = self.model.predict(self.valid_x)
      self.pred = np.argmax(pred, axis=1)
      self.valid_y_ = np.argmax(self.valid_y, axis=1)

    print(accuracy_score(self.valid_y_, self.pred))
    print(accuracy_score(self.valid_y_, self.pred, normalize=False))
    print(precision_recall_fscore_support(self.valid_y_, self.pred))
    print(
        precision_recall_fscore_support(self.valid_y_,
                                        self.pred,
                                        average='micro'))

  def consolidateResult(self, filepath=None):
    import pandas as pd
    df = pd.DataFrame(list(zip(self.pred, self.valid_y_)),
                      columns=['Pred', 'GroundTruth'])
    # df.to_csv(filepath + '/result.csv')

    print(df)


if __name__ == '__main__':
  train_path = 'data/ntuc_train.json'
  valid_path = 'data/ntuc_valid.json'

  dp = DataProcessor(train_path, valid_path)
  dp.processText()  # 4 empties
  dp.getDistinctLabel()
  dp.oneHotLabels()
  dp.randomizeData()

  vc = VanillaConveRT(dp.train_sents, dp.train_labels_onehot, dp.valid_sents,
                      dp.valid_labels_onehot)
  vc.createModel()
  vc.train(filepath='test')
  vc.consolidateResult()
  pdb.set_trace()
"""
  # Internally it implements caching, deduplication, and batching, to help speed up encoding. Note that because it does batching internally, you can pass very large lists of sentences to encode without going out of memory.
client = encoder_client.EncoderClient(
    "http://models.poly-ai.com/convert/v1/model.tar.gz")


context_encodings = client.encode_sentences(train_sents)

print(context_encodings.shape)
"""