import tensorflow as tf

l = [True, False]
t = tf.convert_to_tensor(l, dtype=tf.bool)
print(0.99 * tf.where(t, 1.0, 0.0))