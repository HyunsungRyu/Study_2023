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

print("Logistic regression")

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_train = [[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0], [5.0, 3.0], [6.0, 2.0]]
y_train = [[0.0], [0.0], [0.0], [1.0], [1.0], [1.0]]

x_test = [[5.0, 2.0]]
y_test = [[1.0]]

x1 = [x[0] for x in x_train]  # 1, 2, 3, ...
x2 = [x[1] for x in x_train]  # 2, 3, 1 ...

colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1, x2, c=colors, marker="^")
plt.scatter(x_test[0][0], x_test[0][1], c="red")

plt.xlabel("x1")
plt.xlabel("x2")
plt.show()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))

W = tf.Variable(tf.zeros([2, 1]), name="Weight")
b = tf.Variable(tf.zeros([1]), name="bias")


def logistic_regression(features):
    hypothesis = tf.divide(1.0, 1.0 + tf.exp(-tf.matmul(features, W) + b))
    return hypothesis


def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(1 - hypothesis))
    return cost


optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)


def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy


def grad(features, labels):
    with tf.GradientTape() as tape:
        hypothesis = logistic_regression(features)
        loss_value = loss_fn(hypothesis, labels)
    return tape.gradient(loss_value, [W, b])


EPOCHS = 1001
for step in range(EPOCHS):
    for features, labels in iter(dataset.batch(len(x_train))):
        hypothesis = logistic_regression(features)
        grads = grad(features, labels)
        optimizer.apply_gradients(grads_and_vars=zip(grads, [W, b]))
        if step % 100 == 0:
            print(f"Iter:{step}, Loss: {loss_fn(hypothesis, labels):.4f}")

test_acc = accuracy_fn(logistic_regression(x_test), y_test)
print(f"Test Result = {tf.cast(logistic_regression(x_test) > 0.5, dtype=tf.int32)}")
print(f"Testset Accuracy: {test_acc:.4f}")
