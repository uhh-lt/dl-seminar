# -*- coding: utf-8 -*-

from nltk.corpus import brown
import sklearn
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import numpy.random as random

import itertools

random.seed(42)

## Install data by running the following code:
#import nltk
#nltk.download('brown')
#nltk.download('universal_tagset')

def get_data(train_test_size):

    ## Take a subset
    brown_words = list(itertools.islice(brown.words(), train_test_size))
    brown_tags = [pair[1] for pair in brown.tagged_words(tagset='universal')]

    word_encoder = sklearn.preprocessing.LabelEncoder()
    pos_encoder = sklearn.preprocessing.LabelEncoder()

    brown_words_num = word_encoder.fit_transform(brown_words)
    brown_tags_num = pos_encoder.fit_transform(brown_tags)

    input_dim = len(word_encoder.classes_)
    output_dim = len(pos_encoder.classes_)

    return word_encoder, pos_encoder, brown_words_num, brown_tags_num, input_dim, output_dim

def prepare_data(left_context_len, right_context_len, train_test_size):

    _, _, x_data, y_data, input_dim, output_dim = get_data(train_test_size)

    train_data = [(x_data[i-left_context_len:i+right_context_len+1], y_data[i]) for i in range(left_context_len, len(x_data)-right_context_len)]
    x_train = np.array([pair[0] for pair in train_data])
    y_train = np.array([pair[1] for pair in train_data])

    return input_dim, output_dim, x_train, y_train

# can be used instead of prepare_data to get training data that is split on the sentence level
def prepare_data_sentences(train_test_size):

    word_encoder, pos_encoder, brown_words_num, brown_tags_num, input_dim, output_dim = get_data(train_test_size)

    x_data_sents,y_data_sents = [],[]
    x_data_sent, y_data_sent = [],[]

    dot_label = word_encoder.transform(['.'])[0]
    dot_label_tags = pos_encoder.transform(['.'])[0]

    #split on sentences
    for word,tag in zip(brown_words_num, brown_tags_num):

        if word == dot_label and tag == dot_label_tags:
            if len(x_data_sent) > 0:
                x_data_sents.append(x_data_sent)
                y_data_sents.append(y_data_sent)
                x_data_sent, y_data_sent = [],[]

        x_data_sent.append(word)
        y_data_sent.append(tag)

    return input_dim, output_dim, x_data_sents, y_data_sents
    

# seq_len (int), input_dim (int), output_dim (int), embedding_dim (int), learning_rate (float)
def build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate, num_hidden=50):
    ## input
    x = tf.placeholder(tf.int32, (None, seq_len))
    y = tf.placeholder(tf.int32, (None))
    
    embeddings = tf.Variable(tf.random_uniform([input_dim, embedding_dim], -1.0, 1.0))
    
    ## embedd input
    x_embedd = tf.nn.embedding_lookup(embeddings, x)
    
    cell = tf.contrib.rnn.LSTMCell(num_hidden)
    
    sequence = [x_embedd[:,num] for num in range(seq_len)]
    outputs, state = tf.nn.static_rnn(cell, sequence, dtype=tf.float32, sequence_length=[seq_len]) 
    output = outputs[-1]
    
    ## linear model
    W = tf.Variable(tf.random_uniform([num_hidden, output_dim], -0.01, 0.01, dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([output_dim], -0.01, 0.01, dtype=tf.float32))
    pred = tf.matmul(output, W) + b
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
    
    pred_argmax = tf.argmax(tf.nn.softmax(pred), axis=1)

    ## define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # function must return x (placeholder), y (placeholder),  optimizer (optimizer op),
    # loss (loss op),  pred_argmax (argmax of logits as tensor)
    return x, y, optimizer, loss, pred_argmax

def main():

    # model size parameters
    left_context_len = 4
    right_context_len = 0
    
    # set this higher to get a better model
    train_test_size = 10000
    embedding_dim = 100

    ## Hyperparemeters: experiment with these, too
    learning_rate = 0.01
    epochs = 5

    seq_len = left_context_len + 1 + right_context_len    
    input_dim, output_dim, x_data, y_data = prepare_data(left_context_len, right_context_len, train_test_size)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    x, y, optimizer, loss, pred_argmax = build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate)

    ## start the session
    with tf.Session() as sess:

        ## initalize parameters
        sess.run(tf.global_variables_initializer())
        train_dict = {x: x_train, y: y_train}
        test_dict = {x: x_test, y: y_test}

        print("Initial training loss: " + str(sess.run(loss, train_dict)))
        print("Initial test loss: " + str(sess.run(loss, test_dict)))
 

        for i in range(epochs):
            ## run the optimizer
            epoch_data = list(zip(x_train, y_train))
            np.random.shuffle(epoch_data)
            for x_sample,y_sample in epoch_data:
                train_dict_local = {x: [x_sample], y: [y_sample]}
                loss_val , _ = sess.run([loss, optimizer], train_dict_local)            
            print("Training loss after epoch " + str(i+1) + ":" + str(sess.run(loss, train_dict)), 'last example:', loss_val)

        print("Test loss after training:" + str(sess.run(loss, test_dict)))
        y_pred_test = sess.run(pred_argmax, {x: x_test})

        test_acc = sklearn.metrics.accuracy_score(y_test, y_pred_test)

        print("Test accuracy:", test_acc)

if __name__ == "__main__":
    main()

