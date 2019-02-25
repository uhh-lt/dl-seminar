#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 13:34:30 2017

@author: milde
"""

import tensorflow as tf
import numpy as np

#some random test data
a_data = np.random.rand(256)
b_data = np.random.rand(256)

#construct the graph
a = tf.placeholder(tf.float32, [256])
b = tf.placeholder(tf.float32, [256])
       
x = a+b

print(x)

with tf.device('/cpu'):
    with tf.Session() as sess:
      
       x_data = sess.run(x, {a: a_data, b: b_data})
        
       print(x_data)
        
       #tf.summary.FileWriter("graph_log",sess.graph)