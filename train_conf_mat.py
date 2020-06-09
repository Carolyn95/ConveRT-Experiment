# predict on train set
# save prediction result
# visulize it
from collections import Counter
import re
import os
import pdb
# import spacy
import numpy as np
import pickle as pkl
import tensorflow as tf
import keras.backend as K
import tensorflow_hub as hub
from keras.models import Model
from keras.regularizers import l1
from keras.layers import Input, Lambda, Dense, Dropout, Reshape, BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
import random
import time
from memory_profiler import profile
import pandas as pd

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # remove logging info
import warnings
warnings.filterwarnings('ignore')  # filter out warnings
import pdb


class PredictTrain():

  def __init__(self, sents_path, labels_path):
    self.sents = np.load(sents_path, allow_pickle=True)
    self.labels = np.load(labels_path, allow_pickle=True)
    self.n_labels = len(self.labels[0])
    self.embed = hub.Module('../models/use-module', trainable=False)

  def use_embedding(self, x):
    return self.embed(tf.reshape(tf.cast(x, 'string'), [-1]),
                      signature='default',
                      as_dict=True)['default']

  def createModel(self):
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(self.use_embedding, output_shape=(512,))(input_text)
    dense = Dense(256, activation='relu',
                  kernel_regularizer=l1(0.0001))(embedding)
    dense = Dropout(0.4)(dense)
    pred = Dense(self.n_labels, activation='softmax')(dense)
    self.model = Model(inputs=[input_text], outputs=pred)
    self.model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    print(self.model.summary())

  def createModelBN(self):
    # create model with batch normalization layers
    input_text = Input(shape=(1,), dtype='string')
    embedding = Lambda(self.use_embedding, output_shape=(512,))(input_text)
    dense = Dense(256, activation='relu',
                  kernel_regularizer=l1(0.0001))(embedding)
    dense = BatchNormalization()(dense)
    dense = Dense(256, activation='tanh')(dense)
    # dense = Dropout(0.1)(dense)
    dense = BatchNormalization()(dense)
    pred = Dense(self.n_labels, activation='softmax')(dense)
    self.model = Model(inputs=[input_text], outputs=pred)
    self.model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])
    # opt = Adam(learning_rate=0.001)  # default is 0.001
    # self.model.compile(loss='categorical_crossentropy',
    #                    optimizer=opt,
    #                    metrics=['accuracy'])
    print(self.model.summary())

  def test(self, is_bn, model_path):
    if is_bn:
      self.createModelBN()
      result_path = 'Vanilla_USE_BN/use_bn_mixed_train.csv'
    else:
      self.createModel()
      result_path = 'Vanilla_USE/use_mixed_train.csv'
    self.model_path = model_path
    with tf.Session() as sess:
      K.set_session(sess)
      sess.run(tf.global_variables_initializer())
      sess.run(tf.tables_initializer())
      self.model.load_weights(self.model_path)
      pred = self.model.predict(self.sents)
      self.pred = np.argmax(pred, axis=1)
      self.actual = np.argmax(self.labels, axis=1)
    print(accuracy_score(self.actual, self.pred))
    print(
        precision_recall_fscore_support(self.actual, self.pred,
                                        average='macro'))

    df = pd.DataFrame(list(zip(self.pred, self.actual)),
                      columns=['Pred', 'GroundTruth'])
    df.to_csv(result_path)


if __name__ == '__main__':
  sents_path = 'data/train_sents_mixed.npy'
  labels_path = 'data/train_labels_onehot_mixed.npy'
  pt = PredictTrain(sents_path, labels_path)
  # 'Vanilla_USE/20.hdf5' | 'Vanilla_USE_BN/15.hdf5'
  pt.test(is_bn=True, model_path='Vanilla_USE_BN/15.hdf5')
