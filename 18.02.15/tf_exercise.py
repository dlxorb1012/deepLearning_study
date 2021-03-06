import tensorflow as tf
import numpy as np
'''
f_org = open('filepath', 'r')
f_new = open('filepath', 'w')
while True:
    line = f_org.readline()
    if not line: break
    line = line.replace('x', '1').replace('positive', '1').replace('b', '0.5').replace('o', '-1').replace('negative', '0')
    f_new.write(line)
    print(line)

f_org.close()
f_new.close()
'''
xy = np.loadtxt('filepath', delimiter=',', dtype=np.float32)

x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]


X = tf.placeholder(tf.float32, shape=[None, 9])
Y = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([9, 27]), name='weight1')
w2 = tf.Variable(tf.random_normal([27, 27]), name='weight2')
w3 = tf.Variable(tf.random_normal([27, 1]), name='weight3')
b1 = tf.Variable(tf.random_normal([1]), name='bias1')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
b3 = tf.Variable(tf.random_normal([1]), name='bias3')

a1 = tf.tanh(tf.matmul(X, w1) + b1)
a2 = tf.tanh(tf.matmul(a1, w2) + b2)
hypothesis = tf.sigmoid(tf.matmul(a2, w3) + b3)
cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(3001):
    cost_val, hy_val = sess.run([cost, train], feed_dict={X: x_data, Y: y_data})

    if step % 100 == 0:
        print(step, "Cost: ", cost_val)

h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
print('\nHypothesis: ', h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)

x_input = 0

while True:
    print("player 1: 1, player 2: -1, empty: 0")
    x_input = input('Input your game result: ').split(',')
    x_list = list(map(float, x_input))
    if len(x_list) != 9:
        print('Wrong data type. pls write again')
        continue
    x_list = [x_list]
    answer = sess.run(hypothesis, feed_dict={X: x_list})
    result = 'Win' if int(answer + 0.5) else 'Lose'
    print('prediction: ', answer, '\nResult: ', result)
