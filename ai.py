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


# print("Cost function in pure Python")
# import numpy as np

# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])


# def cost_func(W, X, Y):
#     c = 0
#     for i in range(len(X)):
#         c += W * X[i] - Y[i] ** 2
#     return c / len(X)


# for feed_W in np.linspace(-3, 5, num=15): # -3~5 중 15개의 구간으로 나눔
#     curr_cost = cost_func(feed_W, X, Y)
#     print(f"{feed_W:6.3f} | {curr_cost:10.5f}")

# print("______________________________________")

# print("Cost function in TensorFlow")
# import numpy as np
# import tensorflow as tf

# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])


# def cost_func(W, X, Y):
#     hypothesis = X * W
#     return tf.reduce_mean(tf.square(hypothesis - Y))


# W_values = np.linspace(-3, 5, num=15)
# cost_values = []

# for feed_W in W_values:
#     curr_cost = cost_func(feed_W, X, Y)
#     cost_values.append(curr_cost)
#     print(f"{feed_W:6.3f} | {curr_cost:10.5f}")

# print("______________________________________")

# print("Gradient descent")
# import tensorflow as tf

# alpha = 0.01
# gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
# descent = W - tf.multiply(alpha, gradient)
# W.assign(descent)

# print("______________________________________")

# print("Gradient descent ")
# import tensorflow as tf
# import numpy as np

# tf.random.set_seed(0)  # for reproducibility # tf.set_random_seed(0) 에러

# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])

# x_data = [1.0, 2.0, 3.0, 4.0]
# y_data = [1.0, 3.0, 5.0, 7.0]

# W = tf.Variable(tf.random.normal([1], -100.0, 100.0))

# for step in range(300):
#     hypothesis = W * X
#     cost = tf.reduce_mean(tf.square(hypothesis - Y))

#     alpha = 0.01
#     gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
#     descent = W - tf.multiply(alpha, gradient)
#     W.assign(descent)

#     if step % 10 == 0:
#         print(f"{step:5}{cost.numpy():10.4f}{W.numpy()[0]:10.6f}")

# print("______________________________________")

# print("Gradient descent/Output when W = 5")
# import tensorflow as tf
# import numpy as np

# tf.random.set_seed(0)  # for reproducibility # tf.set_random_seed(0) 에러

# X = np.array([1, 2, 3])
# Y = np.array([1, 2, 3])

# x_data = [1.0, 2.0, 3.0, 4.0]
# y_data = [1.0, 3.0, 5.0, 7.0]

# W = tf.Variable([5.0])

# for step in range(300):
#     hypothesis = W * X
#     cost = tf.reduce_mean(tf.square(hypothesis - Y))

#     alpha = 0.01
#     gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, X) - Y, X))
#     descent = W - tf.multiply(alpha, gradient)
#     W.assign(descent)

#     if step % 10 == 0:
#         print(f"{step:5}{cost.numpy():10.4f}{W.numpy()[0]:10.6f}")

# print("______________________________________")

# print("Hypothesis using matrix")

# import tensorflow as tf
# import numpy as np

# # data and Label
# x1 = [73.0, 93.0, 89.0, 96.0, 73.0]
# x2 = [80.0, 88.0, 91.0, 98.0, 66.0]
# x3 = [75.0, 93.0, 90.0, 100.0, 70.0]
# Y = [152.0, 185.0, 180.0, 196.0, 142]

# # weights
# w1 = tf.Variable(10.0)
# w2 = tf.Variable(10.0)
# w3 = tf.Variable(10.0)
# b = tf.Variable(10.0)

# hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b

# print("______________________________________")

# print("Hypothesis using matrix")

# import tensorflow as tf
# import numpy as np

# # data and Label
# x1 = [73.0, 93.0, 89.0, 96.0, 73.0]
# x2 = [80.0, 88.0, 91.0, 98.0, 66.0]
# x3 = [75.0, 93.0, 90.0, 100.0, 70.0]
# Y = [152.0, 185.0, 180.0, 196.0, 142]

# # random weights
# w1 = tf.Variable(tf.random.normal([1]))
# w2 = tf.Variable(tf.random.normal([1]))
# w3 = tf.Variable(tf.random.normal([1]))
# b = tf.Variable(tf.random.normal([1]))

# learning_rate = 0.000001

# for i in range(1000 + 1):
#     # tf.GradientTape() to record the gradient of the cost function
#     with tf.GradientTape() as tape:
#         hypothesis = w1 * x1 + w2 * x2 + w3 * x3 + b
#         cost = tf.reduce_mean(tf.square(hypothesis - Y))
#     # calculates the gradients of the cost
#     w1_grad, w2_grad, w3_grad, b_grad = tape.gradient(cost, [w1, w2, w3, b])

#     # update w1, w2, w3 and b
#     w1.assign_sub(learning_rate * w1_grad)
#     w2.assign_sub(learning_rate * w2_grad)
#     w3.assign_sub(learning_rate * w3_grad)
#     b.assign_sub(learning_rate * b_grad)

#     if i % 50 == 0:
#         print(f"{i:5} | {cost.numpy():12.4f}")

# print("______________________________________")

# print("Matrix")

# import tensorflow as tf
# import numpy as np

# data = np.array(
#     [
#         [73.0, 80.0, 75.0, 152.0],
#         [93.0, 88.0, 93.0, 185.0],
#         [89.0, 91.0, 90.0, 180.0],
#         [96.0, 98.0, 100.0, 196.0],
#         [73.0, 66.0, 70.0, 142.0],
#     ],
#     dtype=np.float32,
# )

# # slice data
# X = data[:, :-1]
# y = data[:, [-1]]

# W = tf.Variable(tf.random.normal([3, 1]))
# b = tf.Variable(tf.random.normal([1]))

# learning_rate = 0.000001


# # hypothesis, prediction function
# def predict(X):
#     return tf.matmul(X, W) + b


# n_epochs = 2000
# for i in range(n_epochs + 1):
#     with tf.GradientTape() as tape:
#         cost = tf.reduce_mean((tf.square(predict(X) - y)))

#     # calculates the Gradients of the loss
#     W_grad, b_grad = tape.gradient(cost, [W, b])

#     # updates parameters (W and b)
#     W.assign_sub(learning_rate * W_grad)
#     b.assign_sub(learning_rate * b_grad)

#     if i % 100 == 0:
#         print(f"{i:5} | {cost.numpy():10.4f}")

# print("______________________________________")

# print("Sginmoid(Logistic) function")

# hypothesis = tf.sigmoid(z) z = tf.matmul(X, θ) + b
# hypothesis = tf.div(1., 1. + tf.exp(z))

# print("______________________________________")

# print("Cost Function(the cost function to fit the parameters(θ{w}))")

# import tensorflow as tf
# import numpy as np

# def loss_fn(hypothesis, label):
#     cost = -tf.reduce_mean(labels * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))
#     return cost

# print("______________________________________")

# print("Cost Function(A convex logistic regression cost function)")


# import tensorflow as tf
# import numpy as np

# cost = -tf.reduce_mean(labels * tf.log(hypothesis) + (1 - labels) * tf.log(1 - hypothesis))

# print("______________________________________")

# print("Optimization(How to minimize the cost function)")


# import tensorflow as tf
# import numpy as np

# def grad(hypothesis, labels):
#     with tf.GradientTape() as tape:
#         loss_value = loss_fn(hypothesis, labels)
#     return tape.gradient(loss_value, [W, b])
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))

# print("______________________________________")

# print("Logistic regression")

# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

# x_train = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0], [5.0, 3.0], [6.0, 2.0]]
# y_train = [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]

# x_test = [[5.0, 2.0]]
# y_test = [[1.0]]

# x1 = [x[0] for x in x_train]  # 1, 2, 3, ...
# x2 = [x[1] for x in x_train]  # 2, 3, 1 ...

# colors = [int(y[0] % 3) for y in y_train]
# plt.scatter(x1, x2, c=colors, marker="^")
# plt.scatter(x_test[0][0], x_test[0][1], c="red")

# plt.xlabel("x1")
# plt.xlabel("x2")
# plt.show()

# print("______________________________________")

# print("Logistic regression")

# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf

# x_train = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0], [5.0, 3.0], [6.0, 2.0]]
# y_train = [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]

# x_test = [[5.0, 2.0]]
# y_test = [[1.0]]

# x1 = [x[0] for x in x_train]  # 1, 2, 3, ...
# x2 = [x[1] for x in x_train]  # 2, 3, 1 ...

# colors = [int(y[0] % 3) for y in y_train]
# plt.scatter(x1, x2, c=colors, marker="^")
# plt.scatter(x_test[0][0], x_test[0][1], c="red")

# plt.xlabel("x1")
# plt.xlabel("x2")
# plt.show()

# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

# W = tf.Variable(tf.zeros([2, 1]), name="Weight")
# b = tf.Variable(tf.zeros([1]), name="bias")


# def logistic_regression(features):
#     hypothesis = tf.divide(1.0, 1.0 + tf.exp(-tf.matmul(features, W) + b))
#     return hypothesis


# def loss_fn(hypothesis, labels):
#     cost = -tf.reduce_mean(labels * tf.math.log(1 - hypothesis))
#     return cost


# optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


# def accuracy_fn(hypothesis, labels):
#     predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
#     accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
#     return accuracy


# def grad(features, labels):
#     with tf.GradientTape() as tape:
#         hypothesis = logistic_regression(features)
#         loss_value = loss_fn(hypothesis, labels)
#     return tape.gradient(loss_value, [W, b])


# EPOCHS = 1001
# for step in range(EPOCHS):
#     for features, labels in iter(dataset.batch(len(x_train))):
#         hypothesis = logistic_regression(features)
#         grads = grad(features, labels)
#         optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
#         if step % 100 == 0:
#             print(f"Iter:{step}, Loss: {loss_fn(hypothesis, labels):.4f}")

# test_acc = accuracy_fn(logistic_regression(x_test), y_test)
# print(f"Test Result = {tf.cast(logistic_regression(x_test) > 0.5, dtype=tf.int32)}")
# print(f"Testset Accuracy: {test_acc:.4f}")

# print("______________________________________")

# print("Sample Dataset")

# import numpy as np

# x_data = [
#     [1, 2, 1, 1],
#     [2, 1, 3, 2],
#     [3, 1, 3, 4],
#     [4, 1, 5, 5],
#     [1, 7, 5, 5],
#     [1, 2, 5, 6],
#     [1, 6, 6, 6],
#     [1, 7, 7, 7],
# ]
# y_data = [
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [1, 0, 0],
# ]
# # cnvert into numpy and float format
# x_data = np.asarray(x_data, dtype=np.float32)
# y_data = np.asarray(y_data, dtype=np.float32)

# nb_classes = 3  # num classes

# print("______________________________________")

# print("Softmax function")

# import tensorflow as tf

# hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

# tf.matmul(X,W)+b

# print("______________________________________")

# print("Softmax function")

# import tensorflow as tf
# import numpy as np


# W = tfe.Variable(tf.random.normal([4, nb_classes]), name="weight")
# b = tfe.Variable(tf.random.normal([nb_classes]), name="bias")
# variables = [W, b]

# print("______________________________________")

# print("Softmax function")

# import tensorflow as tf
# import numpy as np

# # Sample Dataset
# x_data = [
#     [1, 2, 1, 1],
#     [2, 1, 3, 2],
#     [3, 1, 3, 4],
#     [4, 1, 5, 5],
#     [1, 7, 5, 5],
#     [1, 2, 5, 6],
#     [1, 6, 6, 6],
#     [1, 7, 7, 7],
# ]
# y_data = [
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 0, 1],
#     [0, 1, 0],
#     [0, 1, 0],
#     [0, 1, 0],
#     [1, 0, 0],
#     [1, 0, 0],
# ]

# # convert into numpy and float format
# x_data = np.asarray(x_data, dtype=np.float32)
# y_data = np.asarray(y_data, dtype=np.float32)

# # num classes
# nb_classes = 3

# # hypothesis = tf.nn.softmax(tf.matmul(X,W)+b)

# # Weight and bias setting
# W = tf.Variable(tf.random.normal([4, nb_classes]), name="weight")
# b = tf.Variable(tf.random.normal([nb_classes]), name="bias")

# hypothesis = tf.nn.softmax(tf.matmul(x_data, W) + b)

# # softmax onehot test
# sample_db = [[8, 2, 1, 4]]
# sample_db = np.asarray(sample_db, dtype=np.float32)

# # Cost function: cross entropy
# # Cross entropy cost/Loss
# cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), axis = 1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate= 0.1).minimize(cost)

# # Cost Function
# def cost_fn(X, Y):
#     logits = hypothesis(X)
#     cost = -tf.reduce_sum(Y * tf.log(logits), axis =1)
#     cost_mean = tf.reduce_mean(cost)
#     return cost_mean
# print(cost_fn(x_data, y_data))

"""
# 혼자 공부하는 머신러닝+딥러닝 3강

# 예시 데이터 리스트
bream_length = [
    25.4,
    26.3,
    26.5,
    29.0,
    29.0,
    29.7,
    30.0,
    30.0,
    30.7,
    31.0,
    31.0,
    31.5,
    32.0,
    32.0,
    33.0,
    33.0,
    33.5,
    33.5,
    34.0,
    34.0,
    34.5,
    35.0,
    35.0,
    35.0,
    36.0,
    37.0,
    38.5,
    38.5,
    38.5,
    41.0,
    41.0,
]
bream_weight = [
    242.0,
    290.0,
    340.0,
    363.0,
    430.0,
    450.0,
    500.0,
    390.0,
    450.0,
    500.0,
    475.0,
    500.0,
    500.0,
    340.0,
    600.0,
    600.0,
    700.0,
    700.0,
    610.0,
    650.0,
    575.0,
    685.0,
    620.0,
    680.0,
    700.0,
    725.0,
    720.0,
    714.0,
    850.0,
    1000.0,
    920.0,
]
smelt_length = [
    9.8,
    10.5,
    10.6,
    11.0,
    11.2,
    11.3,
    11.8,
    11.8,
    12.0,
    12.2,
    12.4,
    13.0,
    14.3,
    15.0,
]
smelt_weight = [
    6.7,
    7.5,
    7.0,
    9.7,
    9.8,
    8.7,
    10.0,
    9.9,
    9.8,
    12.2,
    13.4,
    12.2,
    19.7,
    19.9,
]

# 산점도
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)

plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# x와 y 데이터 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 리스트 내포
fish_data = [[l, w] for l, w in zip(length, weight)]

# 정답 준비 | 정답이 1 정답이 아니면 0
fish_target = [1] * 31 + [0] * 14

# k-최근접 이웃
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()  # 기본 타깃 5개
kn.fit(fish_data, fish_target)
print(kn.score(fish_data, fish_target))
print(kn.predict([[30, 600]]))  # array([1]) | 도미

kn49 = KNeighborsClassifier(n_neighbors=45)
kn49.fit(fish_data, fish_target)
print(kn49.score(fish_data, fish_target))
print(kn49.predict([[30, 600]]))  # array([1]) | 도미
""" """
# 혼자 공부하는 머신러닝+딥러닝 4강
# 예시 데이터 리스트
bream_length = [
    25.4,
    26.3,
    26.5,
    29.0,
    29.0,
    29.7,
    30.0,
    30.0,
    30.7,
    31.0,
    31.0,
    31.5,
    32.0,
    32.0,
    33.0,
    33.0,
    33.5,
    33.5,
    34.0,
    34.0,
    34.5,
    35.0,
    35.0,
    35.0,
    36.0,
    37.0,
    38.5,
    38.5,
    38.5,
    41.0,
    41.0,
]
bream_weight = [
    242.0,
    290.0,
    340.0,
    363.0,
    430.0,
    450.0,
    500.0,
    390.0,
    450.0,
    500.0,
    475.0,
    500.0,
    500.0,
    340.0,
    600.0,
    600.0,
    700.0,
    700.0,
    610.0,
    650.0,
    575.0,
    685.0,
    620.0,
    680.0,
    700.0,
    725.0,
    720.0,
    714.0,
    850.0,
    1000.0,
    920.0,
]
smelt_length = [
    9.8,
    10.5,
    10.6,
    11.0,
    11.2,
    11.3,
    11.8,
    11.8,
    12.0,
    12.2,
    12.4,
    13.0,
    14.3,
    15.0,
]
smelt_weight = [
    6.7,
    7.5,
    7.0,
    9.7,
    9.8,
    8.7,
    10.0,
    9.9,
    9.8,
    12.2,
    13.4,
    12.2,
    19.7,
    19.9,
]

# 산점도
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)

plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# x와 y 데이터 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 리스트 내포
fish_data = [[l, w] for l, w in zip(length, weight)]

# 정답 준비 | 정답이 1 정답이 아니면 0
fish_target = [1] * 31 + [0] * 14

# 넘파이 사용하기
import numpy as np

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
print(input_arr)

# 데이터 섞기
index = np.arange(45)
print(index)
np.random.shuffle(index)
print(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

# 데이터 나누고 확인하기
import matplotlib.pyplot as plt

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# 두 번째 머신러닝 프로그램
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()  # 기본 타깃 5개
kn = kn.fit(train_input, train_target)
kn.score(test_input, test_target)
""" """
# print("혼자 공부하는 머신러닝+딥러닝 5강")
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import numpy as np

# print("예시 데이터 리스트")
bream_length = [
    25.4,
    26.3,
    26.5,
    29.0,
    29.0,
    29.7,
    30.0,
    30.0,
    30.7,
    31.0,
    31.0,
    31.5,
    32.0,
    32.0,
    33.0,
    33.0,
    33.5,
    33.5,
    34.0,
    34.0,
    34.5,
    35.0,
    35.0,
    35.0,
    36.0,
    37.0,
    38.5,
    38.5,
    38.5,
    41.0,
    41.0,
]
bream_weight = [
    242.0,
    290.0,
    340.0,
    363.0,
    430.0,
    450.0,
    500.0,
    390.0,
    450.0,
    500.0,
    475.0,
    500.0,
    500.0,
    340.0,
    600.0,
    600.0,
    700.0,
    700.0,
    610.0,
    650.0,
    575.0,
    685.0,
    620.0,
    680.0,
    700.0,
    725.0,
    720.0,
    714.0,
    850.0,
    1000.0,
    920.0,
]
smelt_length = [
    9.8,
    10.5,
    10.6,
    11.0,
    11.2,
    11.3,
    11.8,
    11.8,
    12.0,
    12.2,
    12.4,
    13.0,
    14.3,
    15.0,
]
smelt_weight = [
    6.7,
    7.5,
    7.0,
    9.7,
    9.8,
    8.7,
    10.0,
    9.9,
    9.8,
    12.2,
    13.4,
    12.2,
    19.7,
    19.9,
]

# 산점도
import matplotlib.pyplot as plt

plt.scatter(bream_length, bream_weight)

plt.scatter(smelt_length, smelt_weight)
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# x와 y 데이터 합치기
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 넘파이로 데이터 준비
fish_data = np.column_stack((length, weight))
fish_target = np.concatenate((np.ones(35), np.zeros(10)))

# 사이킷런으로 데이터 나누기
from sklearn.model_selection import train_test_split

train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=43
)
# 수상한 도미
from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)
print(kn.predict([[25, 150]]))

distances, indexes = kn.kneighbors([[25, 150]])

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker="^")
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker="D")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# 기준을 맞춰라
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker="^")
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker="D")
plt.xlim((0, 1000))
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# 표준 점수로 바꾸기
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
print(mean, std)
print(mean, std)

train_scaled = (train_input - mean) / std

# 수상한 도무 다시 표시하기
new = ([25, 150] - mean) / std
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker="^")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()

# 전처리 데이터에서 모델 훈련
kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)

print(kn.predict([new]))

distances, indexes = kn.kneighbors([new])

plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker="^")
plt.scatter(train_scaled[indexes, 0], train_scaled[indexes, 1], marker="D")
plt.xlabel("length")
plt.ylabel("weight")
plt.show()
"""
# # print("혼자 공부하는 머신러닝+딥러닝 6강")

# # 농어의 무게를 예측하라

# import numpy as np

# perch_length = np.array(
#     [
#         8.4,
#         13.7,
#         15.0,
#         16.2,
#         17.4,
#         18.0,
#         18.7,
#         19.0,
#         19.6,
#         20.0,
#         21.0,
#         21.0,
#         21.0,
#         21.3,
#         22.0,
#         22.0,
#         22.0,
#         22.0,
#         22.0,
#         22.5,
#         22.5,
#         22.7,
#         23.0,
#         23.5,
#         24.0,
#         24.0,
#         24.6,
#         25.0,
#         25.6,
#         26.5,
#         27.3,
#         27.5,
#         27.5,
#         27.5,
#         28.0,
#         28.7,
#         30.0,
#         32.8,
#         34.5,
#         35.0,
#         36.5,
#         36.0,
#         37.0,
#         37.0,
#         39.0,
#         39.0,
#         39.0,
#         40.0,
#         40.0,
#         40.0,
#         40.0,
#         42.0,
#         43.0,
#         43.0,
#         43.5,
#         44.0,
#     ]
# )
# perch_weight = np.array(
#     [
#         5.9,
#         32.0,
#         40.0,
#         51.5,
#         70.0,
#         100.0,
#         78.0,
#         80.0,
#         85.0,
#         85.0,
#         110.0,
#         115.0,
#         125.0,
#         130.0,
#         120.0,
#         120.0,
#         130.0,
#         135.0,
#         110.0,
#         130.0,
#         150.0,
#         145.0,
#         150.0,
#         170.0,
#         225.0,
#         145.0,
#         188.0,
#         180.0,
#         197.0,
#         218.0,
#         300.0,
#         260.0,
#         265.0,
#         250.0,
#         250.0,
#         300.0,
#         320.0,
#         514.0,
#         556.0,
#         840.0,
#         685.0,
#         700.0,
#         700.0,
#         690.0,
#         900.0,
#         650.0,
#         820.0,
#         850.0,
#         900.0,
#         1015.0,
#         820.0,
#         1100.0,
#         1000.0,
#         1100.0,
#         1000.0,
#         1000.0,
#     ]
# )
# # 농어의 길이만 사용
# import matplotlib.pyplot as plt

# plt.scatter(perch_length, perch_weight)
# plt.xlabel("length")
# plt.ylabel("weight")
# plt.show()

# # 훈련 세트 준비
# from sklearn.model_selection import train_test_split

# # 훈련 세트와 테스트 세트로 나누자.
# train_input, test_input, train_target, test_target = train_test_split(
#     perch_length, perch_weight, random_state=42
# )
# # 훈련 세트와 테스트 세트를 2차원 배열로 바꾸자.
# train_input = train_input.reshape(-1, 1)
# test_input = test_input.reshape(-1, 1)

# # 회귀 모델 훈련
# from sklearn.neighbors import KNeighborsRegressor

# knr = KNeighborsRegressor()
# knr.fit(train_input, train_target)

# knr.score(train_input, train_target)

# from sklearn.metrics import mean_absolute_error

# test_prediction = knr.predict(test_input)
# mae = mean_absolute_error(test_target, test_prediction)
# print(mae)

# # 과대적함과 과소적함
# knr.score(train_input, train_target)
# knr.score(test_input, test_target)

# # 이웃 개수 줄이기
# # 과대적합 = ↓ 이웃의 개수 ↑ = 과소적합
# knr.n_neighbors = 3
# knr.fit(train_input, train_target)

# print(knr.score(train_input, train_target))

# print(knr.score(test_input, test_target))

# # R^2(score()) = 1 - (타깃-예측)^2의 합 / (타깃 - 평균)^2의 합
# # R^2 = 결정 계수

# # print("혼자 공부하는 머신러닝+딥러닝 7강")

# # 아주 큰 농어
# print(knr.predict([[50]]))

# # 50cm의 농어의 이웃

# # 50cm의 농어의 이웃을 구합니다.
# dictances, indexes = knr.kneighbors([[50]])

# # 훈련 세트의 산섬도를 그립니다.
# plt.scatter(train_input, train_target)
# # 훈련 세트 중에서 이웃 샘플만 다시 그립니다.
# plt.scatter(train_input[indexes], train_target[indexes], marker="D")
# # 50cm 농어 데이터
# plt.scatter(50, 1033, marker="^")
# plt.show()

# # 선형 회귀(linear regression)
# from sklearn.linear_model import LinearRegression

# lr = LinearRegression()
# # 선형 회귀 모델 훈련
# lr.fit(train_input, train_target)

# # 50cm 농어에 대한 예측
# print(lr.predict([[50]]))

# print(lr.coef_, lr.intercept_)

# # 학습한 직선 그리기

# # 훈련 세트의 산점도를 그립니다.
# plt.scatter(train_input, train_target)

# # 15에서 50까지 1차 방적식 그래프를 그립니다.
# plt.plot([15, 50], [15 * lr.coef_ + lr.intercept_, 50 * lr.coef_ + lr.intercept_])

# # 50cm 농어 데이터
# plt.scatter(50, 1241.8, marker="^")
# plt.show()

# print(lr.score(train_input, train_target))

# print(lr.score(test_input, test_target))

# # 다항 회귀

# train_poly = np.column_stack((train_input**2, train_input))
# test_poly = np.column_stack((test_input**2, test_input))

# # 모델 다시 훈련
# lr = LinearRegression()
# lr.fit(train_poly, train_target)

# print(lr.predict([[50**2, 50]]))

# print(lr.coef_, lr.intercept_)

# # 학습한 직선 그리기

# # 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다.
# point = np.arange(15, 50)

# # 훈련 세트의 산점도를 그립니다.
# plt.scatter(train_input, train_target)

# # 15에서 49까지 2차 방정식 그래프를 그립니다.
# plt.plot(point, 1.01 * point**2 - 21.6 * point + 116.05)

# # 50cm 농어 데이터
# plt.scatter([50], [1574], marker="^")
# plt.show()

# print(lr.score(train_poly, train_target))

# print(lr.score(test_poly, test_target))

# # print("혼자 공부하는 머신러닝+딥러닝 8강")

# # 다중 회귀(multiple regression)
# # 판다스로 데이터 준비
# import pandas as pd

# df = pd.read_csv("http://bit.ly/perch_csv")
# perch_full = df.to_numpy()

# print(perch_full)
# from sklearn.preprocessing import PolynomialFeatures

# # degree=2
# poly = PolynomialFeatures()
# poly.fit([[2, 3]])

# # 1(bias), 2, 3, 2**2, 2**3, 3**2
# print(poly.transform([[2, 3]]))

# # LinearRegression
# poly = PolynomialFeatures(include_bias=False)

# poly.fit(train_input)
# train_poly = poly.transform(train_input)

# print(train_poly.shape)
# poly.get_feature_names_out()
# test_poly = poly.transform(test_input)

# from sklearn.linear_model import LinearRegression

# lr = LinearRegression()
# lr.fit(train_poly, train_target)
# print(lr.score(train_poly, train_target))

# print(lr.score(test_poly, test_target))

# # 더 많은 특성 만들기
# poly = PolynomialFeatures(degree=5, include_bias=False)

# poly.fit(train_input)
# train_poly = poly.transform(train_input)
# train_poly = poly.transform(test_input)

# print(train_poly.shape)
# lr.fit(train_poly, train_target)

# print(lr.score(train_poly, train_target))

# print(lr.score(test_poly, test_target))

# # 규제 전에 표준화
# from sklearn.preprocessing import StandardScaler

# ss = StandardScaler()
# ss.fit(train_poly)
# train_scaled = ss.transform(train_poly)
# test_scaled = ss.transfrom(test_poly)

# # 릿지 회귀
# from sklearn.linear_model import Ridge

# ridge = Ridge()
# ridge.fit(train_scaled, train_target)

# print(ridge.score(train_scaled, train_target))

# print(ridge.score(test_scaled, test_target))

# # 적절한 규제 강도 찾기
# train_score = []
# test_score = []
# alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
# for alpha in alpha_list:
#     # 릿지 모델을 만든다.
#     ridge = Ridge(alpha=alpha)
#     # 릿지 모델을 훈련한다.
#     ridge.fit(train_scaled, train_target)
#     # 훈련 점수와 테스트 점수를 저장한다.
#     train_score.append(ridge.score(train_scaled, train_target))
#     test_score.append(ridge.score(test_scaled, test_target))

# plt.plot(np.log10(alpha_list), train_score)
# plt.plot(np.log10(alpha_list), test_score)
# plt.xlabel("alpha")
# plt.ylabel("R^2")
# plt.show()

# ridge = Ridge(alpha=0.1)
# ridge.fit(train_scaled, train_target)

# print(ridge.score(train_scaled, train_target))

# print(ridge.score(test_scaled, test_target))

# # 라쏘 회귀
# from sklearn.linear_model import Lasso

# lasso = Lasso()
# lasso.fit(train_scaled, train_target)

# print(lasso.score(train_scaled, train_target))

# print(lasso.score(test_scaled, test_target))

# lasso = Lasso(alpha=10)
# lasso.fit(train_scaled, train_target)

# print(lasso.score(train_scaled, train_target))

# print(lasso.score(test_scaled, test_target))

# print(np.sum(lasso.coef_ == 0))
"""
[MIT] 데이터 사이언스 기초 강의 
"""


# Class Food
class Food(object):
    def __init__(self, n, v, w):
        self.name = n
        self.value = v
        self.calories = w

    def getValue(self):
        return self.value

    def getCost(self):
        return self.calories

    def density(self):
        return self.getValue() / self.getCost()

    def __str__(self):
        return self.name + ":<" + str(self.value) + "," + str(self.calories) + ">"

    def buildMenu(names, values, calories):
        """names, values, calories lists of same length.
        name a list of strings
        values adn calories lists of numbers
        returns list fo Foods"""
        menu = []
        for i in range(len(values)):
            menu.append(Food(names[i], values[i], calories))
        return menu

    def greedy(items, maxCost, keyFunction):
        """Assumes items a list, maxCost >= 0,
        keyFunction maps elements of items to numbers"""
        itemsCopy = sorted(items, key=keyFunction, reverse=True)
