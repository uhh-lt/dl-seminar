# -*- coding: utf-8 -*-

from nltk.corpus import brown
import sklearn
import tensorflow as tf
import numpy as np
import numpy.random as random

import itertools
import collections

from tensorflow.python.ops import variable_scope

random.seed(42)

## Install data by running the following code:
#import nltk
#nltk.download('brown')
#nltk.download('universal_tagset')

def gen_x_data(x_data, left_context_len, right_context_len):
    return [x_data[i-left_context_len:i+right_context_len+1] for i in range(left_context_len, len(x_data)-right_context_len)]

# tokenized text, word_encoder is a dictionary mapping of words -> ids
def vectorize(text, word_encoder):
    text_ids = []
    for word in text:
        if word in word_encoder:
            text_ids.append(word_encoder[word])
        else:
            text_ids.append(word_encoder['UNK'])
    return np.asarray(text_ids)
            
def prepare_data(left_context_len, right_context_len, training_size, min_frequency=1):
    
    ## Take a subset
    brown_words = list(itertools.islice(brown.words(), training_size))
    brown_tags = [pair[1] for pair in brown.tagged_words(tagset='universal')]
    
    #print(brown_words)

    word_encoder = sklearn.preprocessing.LabelEncoder()
    
    word_counter = collections.defaultdict(int)
    
    brown_words_len = len(brown_words)
    
    brown_words_len_test = int(brown_words_len / 10)
    brown_words_len_train = brown_words_len - brown_words_len_test
    
    print(brown_words_len_train)
    
    brown_words_train = brown_words[:brown_words_len_train]
    brown_words_test = brown_words[brown_words_len_train:]
    
    for word in brown_words_train:
        word_counter[word] += 1
    
    word_counter = list(word_counter.items())
    word_counter = [elem for elem in word_counter if elem[1] >= min_frequency]
    word_counter.sort(key=lambda x:x[1], reverse=True)
    
    word_encoder = {}
    
    # give word ids for all words in vocabulary, most frequent first
    # reserve word for id 0 for UNK, unkown word
    
    word_encoder['UNK'] = 0
    for i,elem in enumerate(word_counter):
        word_encoder[elem[0]] = i+1
    
    pos_encoder = sklearn.preprocessing.LabelEncoder()
    x_data = vectorize(brown_words, word_encoder)
    y_data = pos_encoder.fit_transform(brown_tags)
    
    
    input_dim = len(list(word_encoder.keys()))
    output_dim = len(pos_encoder.classes_)
    
    train_data = [(x_data[i-left_context_len:i+right_context_len+1], y_data[i]) \
                for i in range(left_context_len, len(x_data)-right_context_len)]
    x_data_all = np.array([pair[0] for pair in train_data])
    y_data_all = np.array([pair[1] for pair in train_data])
    
    return word_encoder, pos_encoder, input_dim, output_dim,\
 x_data_all[:brown_words_len_train], y_data_all[:brown_words_len_train],\
 x_data_all[brown_words_len_train:], y_data_all[brown_words_len_train:]

# seq_len (int), input_dim (int), output_dim (int), embedding_dim (int), learning_rate (float)
def build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate):
    ## input
    x = tf.placeholder(tf.int32, (None, seq_len))
    y = tf.placeholder(tf.int32, (None))
    
    dropout_rate = tf.placeholder(tf.float32, ())

    embeddings = tf.Variable(
       tf.random_uniform([input_dim, embedding_dim], -1.0, 1.0))
    
    ## embedd input

    # (None, seq_len, embbeding_dim) -> (None, embbeding_dim * seq_len)
    x_embedd = tf.reshape(tf.nn.embedding_lookup(embeddings, x), [-1, embedding_dim*seq_len])
    
    net = x_embedd
    input_dim = embedding_dim*seq_len
    
    num_layers = 3
    fc_size = 256
    
    # DNN: create 3 hidden layers with a size of 128
    
    for i in range(num_layers):
        with variable_scope.variable_scope("layer%i"%i):
            hidden = tf.Variable(tf.random_uniform([input_dim, fc_size], -0.01, 0.01, dtype=tf.float32), name="hidden")
            b = tf.Variable(tf.random_uniform([fc_size], -0.01, 0.01, dtype=tf.float32), name="b")
    
            net = tf.add(tf.matmul(net, hidden), b)
            net = tf.nn.relu(net)
            
            net = tf.nn.dropout(net, keep_prob=1.0-dropout_rate)
        
        input_dim = fc_size
    
    # output layer, no nonlinearity here
    
    out = tf.Variable(tf.random_uniform([fc_size, output_dim], -0.01, 0.01, dtype=tf.float32))
    b_out = tf.Variable(tf.random_uniform([output_dim], -0.01, 0.01, dtype=tf.float32))
    pred = tf.matmul(net, out) + b_out
    
    # use tensorflows build in softmax + cross entropy
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=pred))
    
    pred_argmax = tf.argmax(tf.nn.softmax(pred), axis=1)

    ## define the optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    # function must return x (placeholder), y (placeholder),  optimizer (optimizer op),
    # loss (loss op),  pred_argmax (argmax of logits as tensor), dropout
    return x, y, dropout_rate, optimizer, loss, pred_argmax

def main():

    # model size parameters
    left_context_len = 2
    right_context_len = 2
    
    # set this higher to get a better model
    training_size = 100000
    embedding_dim = 100

    ## Hyperparemeters: experiment with these, too
    learning_rate = 0.0003
    epochs = 20 #12

    seq_len = left_context_len + 1 + right_context_len    
    word_encoder, pos_encoder, input_dim, output_dim, x_train, y_train, x_test, y_test = prepare_data(left_context_len, right_context_len, training_size)
    x, y, dropout_rate, optimizer, loss, pred_argmax = build_graph(seq_len, input_dim, output_dim, embedding_dim, learning_rate)

    ## start the session
    with tf.Session() as sess:
    
        ## initalize parameters
        sess.run(tf.global_variables_initializer())
        train_dict = {x: x_train, y: y_train, dropout_rate: 0.5}
    
        print("Initial training loss: " + str(sess.run(loss, train_dict)))
    
        for i in range(epochs):
            ## run the optimizer
            epoch_data = list(zip(x_train, y_train))
            np.random.shuffle(epoch_data)
            for x_sample, y_sample in epoch_data:
                train_dict_local = {x: [x_sample], y: [y_sample], dropout_rate: 0.5}
                sess.run(optimizer, train_dict_local)            
            print("Training loss after epoch " + str(i+1) + ":" + str(sess.run(loss, train_dict)))
        
        y_pred_test =  sess.run(pred_argmax, {x: x_test, dropout_rate: 0.0})
        
        print('len x_train:', len(x_train) , 'len x_test:', len(x_test) )
        print('test accuracy:', sklearn.metrics.accuracy_score(y_test, y_pred_test))
        
        y_pred_train =  sess.run(pred_argmax, {x: x_train, dropout_rate: 0.0})
        
        print('train accuracy:', sklearn.metrics.accuracy_score(y_train, y_pred_train))
        
        sen1 = "I have to say , the monthly cost is high . The last".split()
        sen2 = "I have to say , a policemen has always employment because there is always crime . The last".split()

        x_data_1 = vectorize(sen1, word_encoder)
        x_data_2 = vectorize(sen2, word_encoder)

        print(x_data_1)
        res1 = sess.run(pred_argmax, {x: gen_x_data(x_data_1,left_context_len,right_context_len), dropout_rate: 0.0})

        print(x_data_2)
        res2 = sess.run(pred_argmax, {x: gen_x_data(x_data_2,left_context_len,right_context_len), dropout_rate: 0.0})

        print(pos_encoder.classes_)

        pos_decoder = dict(enumerate(pos_encoder.classes_))

        print(sen1[2:])
        print([pos_decoder[elem] for elem in res1])
        print(sen2[2:])
        print([pos_decoder[elem] for elem in res2])


if __name__ == "__main__":
    main()

