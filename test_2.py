"""
* preparing steps
pip install --upgrade mxnet>=1.6.0
pip install gluonnlp
pip install -U spacy
pip install -U spacy-lookups-data
python -m spacy download en_core_web_sm
python -m spacy download en

TODO: explore padding for USE

for now, I will just clip sentences with tokens more than 64
"""
# Preliminary processing
import gluonnlp as nlp
import multiprocessing as mp
import pdb
from rich import print
import time
from mxnet import gluon

tokenizer = nlp.data.SpacyTokenizer('en')
length_clip = nlp.data.ClipSequence(5)

# correct: text = ['can i book a car', 'i need a ride from my place', 'book a taxi']
# wrong: text2 = [['can i book a car'], ['i need a ride from my place'], ['book a taxi']]

text = ['can i book a car', 'i need a ride from my place', 'book a taxi']
train_text = text.copy()
valid_text = text.copy()


def preprocess(sents):
  # Clip the length of email words
  sents = length_clip(tokenizer(sents))
  return sents


def get_length(x):
  return float(len(x))


def preprocess_dataset(dataset):
  start = time.time()

  with mp.Pool() as pool:
    # Each sample is processed in an asynchronous manner.
    dataset = pool.map(
        preprocess, dataset
    )  # => [['can', 'i', 'book', 'a', 'car'], ['i', 'need', 'a', 'ride', 'from'], ['book', 'a', 'taxi']]
    lengths = pool.map(get_length, dataset)  # => [5.0, 5.0, 3.0]
    # dataset = gluon.data.SimpleDataset(pool.map(preprocess, dataset))
    # lengths = gluon.data.SimpleDataset(pool.map(get_length, dataset))
  end = time.time()

  print('Done! Tokenizing Time={:.2f}s, #Sentences={}'.format(
      end - start, len(dataset)))
  return dataset, lengths


data, lengths = preprocess_dataset(text)
train_dataset, train_data_lengths = preprocess_dataset(train_text)
valid_dataset, valid_data_lengths = preprocess_dataset(valid_text)

pdb.set_trace()
"""
# padding

batch_size = 64
bucket_num = 10
bucket_ratio = 0.5


def get_dataloader():

  # Construct the DataLoader Pad data, stack label and lengths
  # batchify_fn = nlp.data.batchify.Tuple(
  #     nlp.data.batchify.Pad(axis=0, pad_val=0), nlp.data.batchify.Stack())

  batchify_fn = nlp.data.batchify.Pad(axis=0, pad_val=0)

  # In this example, we use a FixedBucketSampler,
  # which assigns each data sample to a fixed bucket based on its length.
  batch_sampler = nlp.data.sampler.FixedBucketSampler(train_data_lengths,
                                                      batch_size=batch_size,
                                                      num_buckets=bucket_num,
                                                      ratio=bucket_ratio,
                                                      shuffle=True)
  print(batch_sampler.stats())

  # Training set DataLoader
  train_dataloader = gluon.data.DataLoader(dataset=train_dataset,
                                           batch_sampler=batch_sampler,
                                           batchify_fn=batchify_fn)
  # Validation set DataLoader
  valid_dataloader = gluon.data.DataLoader(dataset=valid_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           batchify_fn=batchify_fn)
  return train_dataloader, valid_dataloader


train_dataloader, valid_dataloader = get_dataloader()

import mxnet as mx
ctx = mx.gpu()

for batch_x, batch_y in train_dataloader:
  batch_x, batch_x2 = batch_x.as_in_context(ctx)
  batch_y, batch_y2 = batch_y.as_in_context(ctx)
  pdb.set_trace()
"""