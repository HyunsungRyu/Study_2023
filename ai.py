# import tensorflow as tf


# x_data = [1, 2, 3, 4, 5]
# y_data = [1, 2, 3, 4, 5]

# W = tf.Variable(2.9)
# b = tf.Variable(0.5)

# # hypothesis = W * x + b
# hypothesis = W * x_data + b

# # cost(W, b)
# cost = tf.reduce_mean(tf.square(hypothesis - y_data))

# # tf.reduce_mean()
# v = [1.0, 2.0, 3.0, 4.0]
# tf.reduce_mean(v)  # 2.5
# # 차원을 줄어들게 함 1차원 -> 0차원


# import tensorflow as tf

# tf.enable_eager_execution()  # 즉시 실행

# # Data
# x_data = [1, 2, 3, 4, 5]
# y_data = [1, 2, 3, 4, 5]

# # W, b initialize
# W = tf.Variable(2.9)  # 값을 임의로 지정
# b = tf.Variable(0.5)  # 값을 임의로 지정

# learning_rate = 0.01

# for i in range(100 + 1):  # W, b update
#     # gradient descent
#     with tf.GradientTape() as tape:
#         typothesis = W * x_data + b
#         cost = tf.reduce_mean(tf.square(hypothesis - y_data))
#     W_grad, b_grad = tape.gradient(cost, [W, b])
#     W.assign_sub(learning_rate * W_grad)
#     b.assign_sub(learning_rate * b_grade)
#     if i % 10 == 0:
#         print("{:5}|{:10.4f}|{10.4}|{:10.6}".format(i, W.numpy(), b, numpy(), cost))


print("Cost function in pure Python")
import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


def cost_func(W, X, Y):
    c = 0
    for i in range(len(X)):
        c += W * X[i] - Y[i] ** 2
    return c / len(X)


for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_W, X, Y)
    print(f"{feed_W:6.3f} | {curr_cost:10.5f}")

print("______________________________________")

print("Cost function in TensorFlow")
import numpy as np
import tensorflow as tf

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])


def cost_func(W, X, Y):
    hypothesis = X * W
    return tf.reduce_mean(tf.square(hypothesis - Y))


W_values = np.linspace(-3, 5, num=15)
cost_values = []

for feed_W in W_values:
    curr_cost = cost_func(feed_W, X, Y)
    cost_values.append(curr_cost)
    print(f"{feed_W:6.3f} | {curr_cost:10.5f}")
