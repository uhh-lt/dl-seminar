# -*- coding: utf-8 -*-

from nltk.corpus import brown
import sklearn
import tensorflow as tf
import numpy as np
import numpy.random as random

import itertools

random.seed(42)

## Install data by running the following code:
#import nltk
#nltk.download('brown')
#nltk.download('universal_tagset')

def prepare_data(left_context_len, right_context_len, training_size):
    
    ## Take a subset
    brown_words = list(itertools.islice(brown.words(), training_size))
    brown_tags = [pair[1] for pair in brown.tagged_words(tagset='universal')]
    
    word_encoder = sklearn.preprocessing.LabelEncoder()
    pos_encoder = sklearn.preprocessing.LabelEncoder()
    x_data = word_encoder.fit_transform(brown_words)
    y_data = pos_encoder.fit_transform(brown_tags)
    
    input_dim = len(word_encoder.classes_)
    output_dim = len(pos_encoder.classes_)
    
    train_data = [(x_data[i-left_context_len:i+right_context_len+1], y_data[i]) for i in range(left_context_len, len(x_data)-right_context_len)]
    x_train = np.array([pair[0] for pair in train_data])
    y_train = np.array([pair[1] for pair in train_data])
    
    return input_dim, output_dim, x_train, y_train, pos_encoder

# seq_len (int), input_dim (int), output_dim (int), embedding_dim (int), learning_rate (float)
def build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate):
    ## input
    x = tf.placeholder(tf.int32, (None, seq_len))
    y = tf.placeholder(tf.int32, (None))
    
    embeddings = tf.Variable(
       tf.random_uniform([input_dim, embedding_dim], -1.0, 1.0))
    
    ## embedd input
    x_embedd = tf.reshape(tf.nn.embedding_lookup(embeddings, x), [-1, embedding_dim*seq_len])
    
    ## multilayer model
    hidden_1 = tf.Variable(tf.random_uniform([embedding_dim*seq_len, embedding_dim], -0.01, 0.01, dtype=tf.float32))
    b1 = tf.Variable(tf.random_uniform([embedding_dim], -0.01, 0.01, dtype=tf.float32))
    
    layer_1 = tf.add(tf.matmul(x_embedd, hidden_1), b1)
    layer_1 = tf.nn.relu(layer_1)
    
    hidden_2 = tf.Variable(tf.random_uniform([embedding_dim, embedding_dim], -0.01, 0.01, dtype=tf.float32))
    b2 = tf.Variable(tf.random_uniform([embedding_dim], -0.01, 0.01, dtype=tf.float32))
    
    layer_2 = tf.add(tf.matmul(layer_1, hidden_2), b2)
    layer_2 = tf.nn.relu(layer_2)
    
    out = tf.Variable(tf.random_uniform([embedding_dim, output_dim], -0.01, 0.01, dtype=tf.float32))
    b_out = tf.Variable(tf.random_uniform([output_dim], -0.01, 0.01, dtype=tf.float32))
    pred = tf.matmul(layer_2, out) + b_out
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
    
    pred_argmax = tf.argmax(tf.nn.softmax(pred), axis=1)

    ## define the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # function must return x (placeholder), y (placeholder),  optimizer (optimizer op),
    # loss (loss op),  pred_argmax (argmax of logits as tensor)
    return x, y, optimizer, loss, pred_argmax

def main():

    # model size parameters
    left_context_len = 2
    right_context_len = 2
    
    # set this higher to get a better model
    training_size = 500
    embedding_dim = 100

    ## Hyperparemeters: experiment with these, too
    learning_rate = 0.01
    epochs = 10

    seq_len = left_context_len + 1 + right_context_len    
    input_dim, output_dim, x_train, y_train, pos_encoder = prepare_data(left_context_len, right_context_len, training_size)
    x, y, optimizer, loss, pred_argmax = build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate)

    ## start the session
    with tf.Session() as sess:
    
        ## initalize parameters
        sess.run(tf.global_variables_initializer())
        train_dict = {x: x_train, y: y_train}
    
        print("Initial training loss: " + str(sess.run(loss, train_dict)))
    
        for i in range(epochs):
            ## run the optimizer
            epoch_data = list(zip(x_train, y_train))
            np.random.shuffle(epoch_data)
            for x_sample,y_sample in epoch_data:
                train_dict_local = {x: [x_sample], y: [y_sample]}
                sess.run(optimizer, train_dict_local)            
            print("Training loss after epoch " + str(i+1) + ":" + str(sess.run(loss, train_dict)))
        
        print(pos_encoder.inverse_transform(sess.run(pred_argmax, train_dict)))

if __name__ == "__main__":
    main()

