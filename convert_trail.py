# --------------------------- experiment convert

# remove logging info
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# filter out warnings
import warnings
warnings.filterwarnings('ignore')

import tensorflow_hub as tfhub
import tensorflow as tf
import tensorflow_text
import pdb

sess = None
assert tf.__version__ == '1.14.0', (
    "Found tf version {tf.__version__}, but need 1.14.0")
assert tf.test.is_gpu_available(), (
    "GPU not available. please use a GPU runtime")

# compute sentence encoding
if sess is not None:
  sess.close()

sess = tf.compat.v1.InteractiveSession(graph=tf.Graph())
module = tfhub.Module("http://models.poly-ai.com/convert/v1/model.tar.gz")
text_placeholder = tf.compat.v1.placeholder(dtype=tf.string, shape=None)
encoding_tensor = module(text_placeholder,
                         signature='encode_sequence',
                         as_dict=True)
encoding_tensor2 = module(text_placeholder, signature='tokenize')
print(type(encoding_tensor))
print(encoding_tensor.keys())
# pdb.set_trace()
# encoding_dim = int(encoding_tensor.shape[1])
# print("ConveRT encodes text to {}-dimentional vectors".format(encoding_dim))
sess.run(tf.compat.v1.tables_initializer())
sess.run(tf.compat.v1.global_variables_initializer())


def encode(text):
  # encode the given text to the encoding space
  return sess.run(encoding_tensor, feed_dict={text_placeholder: text})


text = ['can i book a car', 'i need a ride from my place', 'book a taxi']
encoding = encode(text)
sequence_encoding = encoding['sequence_encoding']
print(sequence_encoding)
print(sequence_encoding.shape)
tokens = encoding['tokens']
print(tokens)
tokenized_encoding = sess.run(encoding_tensor2,
                              feed_dict={text_placeholder: text})
# print(encoding.shape)
print(tokenized_encoding)
if sess is not None:
  sess.close()

# --------------------------- experiment convert encode client
# encode_sentences
import encoder_client
import pdb
# Internally it implements caching, deduplication, and batching, to help speed up encoding. Note that because it does batching internally, you can pass very large lists of sentences to encode without going out of memory.
client = encoder_client.EncoderClient(
    "http://models.poly-ai.com/convert/v1/model.tar.gz")
# find good responses to the folowing context
context_encodings = client.encode_contexts(["What's your name?"])
# rank the following reponses as candidates
candidate_responses = ["No, thanks.", "I'm Matt", "Hey", "I have a dog"]
response_encodings = client.encode_responses(candidate_responses)
# compute score using dot product
scores = response_encodings.dot(context_encodings.T).flatten()
# output top score response
top_idx = scores.argmax()
pdb.set_trace()
print('Best response: {}, score: {:.3f}'.format(candidate_responses[top_idx],
                                                scores[top_idx]))
