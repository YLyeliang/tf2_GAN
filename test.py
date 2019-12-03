import numpy as np
import math
import tensorflow as tf
import numpy
# pred & true
l = np.array([[0., 0., 1., 1.], [1., 1., 1., 0.]])

def sigmoid_loss(x,z):
#    return z * -np.log(sigmoid(x)) + (1 - z) * -np.log(1 - sigmoid(x))
    return np.where(x>0,x,0) - x * z + np.log(1 + np.exp(-abs(x)))
def sigmoid(x):
    return 1/(1+np.exp(x))

print(sigmoid_loss(*l).sum())
bce = tf.keras.losses.BinaryCrossentropy()
b=tf.nn.sigmoid_cross_entropy_with_logits(*l)
print(tf.reduce_mean(b))
loss = bce([0., 0., 1., 1.], [1., 1., 1., 0.])
print('Loss: ', loss.numpy())  # Loss: 11.522857