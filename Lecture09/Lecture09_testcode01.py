# XOR problem

import tensorflow.compat.v1 as tf
import numpy as np

tf.disable_eager_execution()
tf.set_random_seed( 777 )

x_data = np.array( [[0, 0], 
                    [0, 1],
                    [1, 0],
                    [1, 1]], dtype = np.float32)
y_data = np.array( [[0], 
                    [1],
                    [1],
                    [1]], dtype = np.float32)

# Placeholder & Variables
X = tf.placeholder(tf.float32, [None, 2])
Y = tf.placeholder(tf.float32, [None, 1])
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

# Hypothesis & Cost function & Optimizer 
hypothesis = tf.sigmoid( tf.matmul(X, W) + b )
cost = -tf.reduce_mean( Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis) )
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize( cost )

# Accuracy computation
# True if hypothesis > 0.5, else Flase
predicted = tf.cast( hypothesis>0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean( tf.cast( tf.equal(predicted, Y), dtype=tf.float32 ) )

# Launch graph
with tf.Session() as sess:
    sess.run( tf.global_variables_initializer() )

    for step in range(10001):
        _, cost_val, w_val = sess.run( [train, cost, W],
                                        feed_dict={X:x_data, Y:y_data} )
    
        if step % 100 == 0:
            print(step, cost_val, w_val)

    h, p, a = sess.run( [hypothesis, predicted, accuracy],
                        feed_dict={X:x_data, Y:y_data} )
    print("\nHypothesis: ", h, "\nCorrect: ", p, "\nAccuracy: ", a)
