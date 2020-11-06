# CNN Basics

import numpy as np  
import tensorflow as tf 
import matplotlib.pyplot as plt 
import random

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed( 777 )

# Load MNIST data
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0

x_train = x_train.reshape( len(x_train), 784)
x_test = x_test.reshape( len(x_test), 784)

y_train = y_train.reshape( len(x_train), 1)
y_train_onehot = np.zeros( (len(x_train),10), dtype=np.float32 )
for k in range(len(x_train)):
    y_train_onehot[k][y_train[k]] = 1.0

y_test = y_test.reshape( len(x_test), 1)
y_test_onehot = np.zeros( (len(x_test),10), dtype=np.float32 )
for k in range(len(x_test)):
    y_test_onehot[k][y_test[k]] = 1.0

# Hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# Input placeholder
X = tf.compat.v1.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X,[-1, 28, 28, 1])
Y = tf.compat.v1.placeholder(tf.float32, [None, 10])

# 1st layer --> Input image's shape = (?, 28, 28, 1)
# Conv --> (?, 28, 28, 32)
# Pool --> (?, 14, 14, 32)
W1 = tf.Variable(tf.compat.v1.random_normal( [3, 3, 1, 32], 
                                             stddev=0.01))

L1 = tf.nn.conv2d( X_img, 
                   W1, 
                   strides=[1,1,1,1], 
                   padding='SAME' )
L1 = tf.nn.relu( L1 )
L1 = tf.nn.max_pool( L1, 
                     ksize=[1, 2, 2, 1], 
                     strides=[1, 2, 2, 1], 
                     padding='SAME')

# L2 --> Input image's shape = (?, 14, 14, 32)
# Conv --> ?, 14, 14, 64
# Pool --> ?, 7, 7, 64
W2 = tf.Variable(tf.compat.v1.random_normal( [3, 3, 32, 64], 
                                             stddev=0.01))

L2 = tf.nn.conv2d( L1, 
                   W2, 
                   strides=[1,1,1,1], 
                   padding='SAME' )
L2 = tf.nn.relu( L2 )
L2 = tf.nn.max_pool( L2, 
                     ksize=[1, 2, 2, 1], 
                     strides=[1, 2, 2, 1], 
                     padding='SAME')
L2_flat = tf.reshape( L2, [-1, 7*7*64] )

# Final Fully Connected 7*7*64 inputs --> 10 output
W3 = tf.compat.v1.get_variable( "W3",
                                shape=[7*7*64, 10] )
b = tf.Variable(tf.compat.v1.random_normal([10]))
logits = tf.matmul(L2_flat, W3) + b 

# Cost function & Optimizer
cost = tf.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2( logits=logits,
                                                                            labels=Y))
optimizer = tf.compat.v1.train.AdamOptimizer( learning_rate=learning_rate).minimize( cost )

# Launch the graph
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    # Start Training 
    print('<------------ Learning Started!! ------------>')
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int( len(x_train)/batch_size )

        for i in range(total_batch):
            batch_xs, batch_ys = x_train[i*batch_size:(i+1)*batch_size], y_train_onehot[i*batch_size:(i+1)*batch_size]
            feed_dict = {X:batch_xs, Y:batch_ys}
            c, _ = sess.run([cost, optimizer], feed_dict = feed_dict)
            avg_cost += c / total_batch

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('<------------ Learning Finished! ----------->')

    # Test model and check accuracy
    correct_prediction = tf.equal( tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )
    print('Accuracy:', sess.run(accuracy, feed_dict={X:x_test, Y:y_test_onehot, keep_prob: 1}))
