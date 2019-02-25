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

# tokenized text, word_encoder is a dictionary mapping of words -> ids
def vectorize(text, word_encoder):
    text_ids = []
    for word in text:
        if word in word_encoder:
            text_ids.append(word_encoder[word])
        else:
            text_ids.append(word_encoder['UNK'])
    return np.asarray(text_ids)

def get_data(train_test_size):

    ## Take a subset
    brown_words = list(itertools.islice(brown.words(), train_test_size))
    brown_tags = [pair[1] for pair in brown.tagged_words(tagset='universal')]

    # the LabelEncoder is not a good idea for the word encoder,
    # better to build your own encoder as in 04_dnn_tagger_04.py
    # this one can't handle OOVs - bad for testing and using the tagger.
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
def build_graph_static(seq_len, input_dim, output_dim, embedding_dim, learning_rate):
    ## input
    x = tf.placeholder(tf.int32, (None, seq_len))
    y = tf.placeholder(tf.int32, (None))
    
    embeddings = tf.Variable(
       tf.random_uniform([input_dim, embedding_dim], -1.0, 1.0))
    
    ## embedd input
    x_embedd = tf.reshape(tf.nn.embedding_lookup(embeddings, x), [-1, embedding_dim*seq_len])
    
    ## linear model
    W = tf.Variable(tf.random_uniform([embedding_dim*seq_len, output_dim], -0.01, 0.01, dtype=tf.float32))
    b = tf.Variable(tf.random_uniform([output_dim], -0.01, 0.01, dtype=tf.float32))
    pred = tf.matmul(x_embedd, W) + b
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
    
    pred_argmax = tf.argmax(tf.nn.softmax(pred), axis=1)

    ## define the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    
    # function must return x (placeholder), y (placeholder),  optimizer (optimizer op),
    # loss (loss op),  pred_argmax (argmax of logits as tensor)
    return x, y, optimizer, loss, pred_argmax

# input_dim (int), output_dim (int), embedding_dim (int), learning_rate (float)
def build_graph_dyn(input_dim, output_dim, embedding_dim, learning_rate, max_sent_len, num_hidden=50):
    
    return x, y, seq_len_in, optimizer, loss, pred_argmax

def main():

    # model size parameters
    
    # these are not needed any more, since the training input is now a whole sentence and its corresponding labels
    #left_context_len = 2
    #right_context_len = 2
    
    # set this higher to get a better model
    train_test_size = 50000
    embedding_dim = 100

    ## Hyperparemeters: experiment with these, too
    learning_rate = 0.001
    epochs = 5

    #seq_len = left_context_len + 1 + right_context_len    
    #input_dim, output_dim, x_train, y_train = prepare_data(left_context_len, right_context_len, train_test_size)
    input_dim, output_dim, x_data, y_data = prepare_data_sentences(train_test_size)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data)
    
    max_sent_len = 0
    for sent in x_train:
        max_sent_len = max(max_sent_len,len(sent))
    
    print('max_sent_len:', max_sent_len)
    
    print('First 3 entries of x_train:', x_train[:3])
    print('First 3 entries of y_train:', y_train[:3])
    
    x, y, seq_len_in, optimizer, loss, pred_argmax = build_graph_dyn(input_dim, output_dim, embedding_dim, learning_rate, max_sent_len)

    print('y',y)

    ## start the session
    with tf.Session() as sess:
    
        ## initalize parameters
        sess.run(tf.global_variables_initializer())    
    
        for i in range(epochs):
            ## run the optimizer
            epoch_data = list(zip(x_train, y_train))
            np.random.shuffle(epoch_data)
            for x_sample,y_sample in epoch_data:
                train_dict_local = {x: [x_sample], y: [y_sample], seq_len_in: [len(x_sample)]}
                #print('train_dict_local',train_dict_local)
                _, loss_val = sess.run([optimizer,loss], train_dict_local)            
            print("Training loss after epoch " + str(i+1) + ":", loss_val)

        test_losses = []  
        test_accs = []      
        for x_test_sample, y_test_sample in zip(x_test, y_test):
            test_train_dict = {x: [x_test_sample], y: [y_test_sample], seq_len_in: [len(x_test_sample)]}
            y_pred_test_sample = sess.run(pred_argmax, test_train_dict)
            test_acc = sklearn.metrics.accuracy_score(y_test_sample, y_pred_test_sample)
            test_accs.append(test_acc)
            test_loss = sess.run(loss, test_train_dict)
            test_losses.append(test_loss)

        print('Mean test loss:', np.mean(test_losses))

        # with the default settings:
        # Mean test accuracy: 0.898555251481
        print('Mean test accuracy:', np.mean(test_accs))

if __name__ == "__main__":
    main()

