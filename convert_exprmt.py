import tensorflow_hub as tfhub
import tensorflow as tf
import tensorflow_text

sess = None
assert tf.__version__ == '1.14.0', (
    "Found tf version {tf.__version__}, but need 1.14.0")
assert tf.test.is_gpu_available(), (
    "GPU not available. please use a GPU runtime")
