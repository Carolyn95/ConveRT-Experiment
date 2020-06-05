# padding by 64 tokens per sentence
# batch norm?
#
from tf.keras.preprocessing.sequence import sequence
import pandas as pd

maxlen = 32
raw_inputs = pd.Series(['this fire fox jump over the dog', 'hello world'])

raw_inputs.str.split(' ').str.len().as_matrix()

total_email = np.load('./processed_data/NoLongEmail_email.npy',
                      allow_pickle=True)
total_email = pd.Series(total_email)

tl = total_email.str.split(' ').str.len().as_matrix()

raw_inputs.str.split(' ').str.len().as_matrix()


def apply_f(a, f):
  if isinstance(a, list):
    return map(lambda t: apply_f(t, f), a)
  else:
    return f(a)


apply_f(total_email, lambda x: np.Series(x).str.split(' ').str.len())

# padded_inputs = sequence.pad_sequences(raw_inputs,
#                                        maxlen=maxlen,
#                                        padding='post')

# MAX_LENGTH = 40

# def filter_max_length(x, y, max_length=MAX_LENGTH):
#   return tf.logical_and(tf.size(x) <= max_length, tf.size(y) <= max_length)
