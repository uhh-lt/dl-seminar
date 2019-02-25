# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

X_data = np.random.rand(500)
Y_data = 2*X_data + 1

X = tf.placeholder(tf.float32, [None])
Y = tf.placeholder(tf.float32, [None])

w = tf.Variable(tf.ones([]))
b = tf.Variable(tf.zeros([]))

yhat = tf.add(tf.multiply(X, w), b)
loss = tf.reduce_mean(tf.square(yhat - Y))

epochs = 1000
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss)

with tf.Session() as sess:
    ## initalize parameters
    sess.run(tf.global_variables_initializer())

    for i in range(epochs):
        ## run one epoch
        sess.run(optimizer, {X: X_data, Y: Y_data})
        ## print result and loss
        print("Epoch " + str(i + 1) + ", Loss: " + str(sess.run(loss, {X: X_data, Y: Y_data})))
    # print(str(sess.run(yhat)))
    print("w: " + str(sess.run(w, {X: X_data, Y: Y_data})))
    print("b: " + str(sess.run(b, {X: X_data, Y: Y_data})))
