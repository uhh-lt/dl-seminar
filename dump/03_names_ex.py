# -*- coding: utf-8 -*-

## http://www.nltk.org/book/ch06.html

import numpy as np
## names must be installed by running (e.g. interactively):
#import nltk
#nltk.download('names')
from nltk.corpus import names

from sklearn.feature_extraction import DictVectorizer

import tensorflow as tf
import numpy.random as random
random.seed(42)

def gender_features(word):
    return {'suffix1': word[-1:],
            'suffix2': word[-2:]}

# input_dim (int)
def build_graph(input_dim):
    ### define the model
    
    # function must return x (placeholder), y (placeholder), pred (tensor), 
    # cost (op), optimizer (op) 
    return x, y, pred, cost, optimizer

if __name__ == "__main__":
    labeled_names = ([(name, 0) for name in names.words('male.txt')] +
    [(name, 1) for name in names.words('female.txt')])
    
    random.shuffle(labeled_names)
    
    featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
    train_set, test_set = featuresets[500:], featuresets[:500]
    
    train_feats = [namefeats[0] for namefeats in train_set]
    train_Y = [namefeats[1] for namefeats in train_set]
    train_Y = np.array(train_Y).reshape(len(train_Y), 1)
    
    test_feats = [namefeats[0] for namefeats in test_set]
    test_Y = [namefeats[1] for namefeats in test_set]
    test_Y = np.array(test_Y).reshape(len(test_Y), 1)
    
    feat_vectorizer = DictVectorizer(dtype=np.int32, sparse=False)
    
    train_X = feat_vectorizer.fit_transform(train_feats)
    test_X = feat_vectorizer.transform(test_feats)
    
    x, y, pred, cost, optimizer =  build_graph(input_dim=test_X.shape[1])
    ## run everything
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # Training cycle
        for epoch in range(5):
            # Run optimization op (backprop) and cost op (to get loss value)
            for x_in, y_out in zip(train_X, train_Y):
                _, c = sess.run([optimizer, cost], feed_dict={x: x_in.reshape(1, len(x_in)),
                                                              y: y_out.reshape(1,1)})
            print(c)
        print("Optimization Finished!")
        print(sess.run(tf.sigmoid(pred), {x: train_X, y: train_Y}))
