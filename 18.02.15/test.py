import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def f(x):
    return 3*(x-2) * 2*(x-3) * (x+2) + 2


X = tf.placeholder(tf.float32, shape=[20])
Y = tf.placeholder(tf.float32, shape=[20])
x_data = np.arange(-8, 12)
y_data = f(x_data)

w1 = tf.Variable(tf.random_normal([1]), name='weight1')
w2 = tf.Variable(tf.random_normal([1]), name='weight2')
w3 = tf.Variable(tf.random_normal([1]), name='weight3')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = (w1 * X*X*X) + (w2 * X*X) + (w3*X) + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000004)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


plt.plot(x_data, y_data)


for step in range(10001):
    hy_val, cost_, _ = sess.run([hypothesis, cost, train], feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
        print(step)
        print(w1)
        plt.plot(x_data, hy_val)

plt.show()