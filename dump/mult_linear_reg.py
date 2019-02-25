# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 11:09:33 2017

@author: barteld
"""

import tensorflow as tf

from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

total_X, total_Y = load_boston(True)
total_X = scale(total_X)
total_Y = total_Y.reshape(len(total_Y), 1)

X = tf.placeholder(tf.float32, [None, 13])
Y = tf.placeholder(tf.float32, [None, 1])

w = tf.Variable(tf.ones([13, 1]))
b = tf.Variable(tf.zeros([]))

yhat = tf.add(tf.matmul(X, w), b)
loss = tf.reduce_mean(tf.square(yhat - Y))

epochs = 100
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(
    learning_rate).minimize(loss)

with tf.Session() as sess:
    ## initalize parameters
    sess.run(tf.global_variables_initializer())

    data_dict = {X: total_X, Y: total_Y}
    for i in range(epochs):
        ## run one epoch
        sess.run(optimizer, data_dict)
        ## print result and loss
        print("Epoch " + str(i + 1) + ", Loss: " + str(sess.run(loss, data_dict)))
    #print(str(sess.run(yhat, data_dict)))