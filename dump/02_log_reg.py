# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

def load_data():
    ## load the data
    bc = load_breast_cancer()
    x_data = bc['data']                                 # shape: (569,30)
    y_data = bc['target'].reshape(len(bc['target']), 1) # shape: (569, 1)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    return x_train, y_train, x_test, y_test


def build_graph(learning_rate):
    ## define the model
    x = tf.placeholder(tf.float64, (None, 30))
    y = tf.placeholder(tf.float64, (None,  1))

    w = tf.Variable(tf.ones([30, 1], dtype=tf.float64), name="weight", dtype=tf.float64)
    b = tf.Variable(tf.zeros([1], dtype=tf.float64), name="bias", dtype=tf.float64)

    yhat = tf.matmul(x, w) + b
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=yhat))

    ## define the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    return x, y, tf.sigmoid(yhat), loss, optimizer

def main():

    ## Hyperparameters
    learning_rate = 0.1
    epochs = 30

    x_train, y_train, x_test, y_test = load_data()
    x, y, pred, loss, optimizer = build_graph(learning_rate)

    ## start the session
    with tf.Session() as sess:

        ## initalize parameters
        sess.run(tf.global_variables_initializer())

        train_dict = {x: x_train, y: y_train}
        test_dict  = {x: x_test,  y: y_test }

        print("Initial training loss: " + str(sess.run(loss, train_dict)))

        for i in range(epochs):
            ## run the optimizer
            sess.run(optimizer, train_dict)
            print("Training loss after epoch " + str(i+1) + ":" + str(sess.run(loss, train_dict)))

        print("Test loss: " + str(sess.run(loss, test_dict)))

if __name__ == "__main__":
    main()
