import tensorflow as tf


x_data = [1, 2, 3, 4, 5]
y_data = [1, 2, 3, 4, 5]

W = tf.Variable(2.9)
b = tf.Variable(0.5)

# hypothesis = W * x + b
hypothesis = W * x_data + b

# cost(W, b)
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
