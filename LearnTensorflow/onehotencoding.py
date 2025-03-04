import tensorflow as tf
import numpy as np

l = [3, 2, 0, 1, 5]
n = np.array(l)
print(n)
print(type(n))
print(n.shape)

one_hot = tf.one_hot(np.squeeze(n).astype(np.float32), depth= 8)
print(one_hot)
