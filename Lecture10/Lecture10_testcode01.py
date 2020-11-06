# MNIST with SOFTMAX

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import random

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed( 777 )

# Load MNIST data
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

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

# placeholder and variables 
X = tf.compat.v1.placeholder( tf.float32, [None, 784] )
Y = tf.compat.v1.placeholder( tf.float32, [None, 10] )

W = tf.Variable( tf.compat.v1.random_normal([784, 10]) )
b = tf.Variable( tf.compat.v1.random_normal([10]) )

# Parameters
learning_rate = 0.001
batch_size = 100
num_epochs = 50
num_examples = len( x_train )
num_iterations = int( num_examples / batch_size )

# Hypothesis & cost function & Optimizer
hypothesis = tf.matmul(X, W) + b

cost = tf.reduce_mean( tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2
                        (logits = hypothesis, labels=tf.stop_gradient(Y)) )

train = tf.compat.v1.train.AdamOptimizer( learning_rate = learning_rate).minimize( cost )

# Accuracy computation
correct_prediction = tf.equal( tf.argmax(hypothesis, axis=1), tf.argmax(Y, axis=1) )
accuracy = tf.reduce_mean( tf.cast(correct_prediction, tf.float32) )

# Train my model
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    #Training cycle
    for epoch in range( num_epochs ):
        avg_cost = 0

        for i in range( num_iterations ):
            batch_xs, batch_ys = x_train[i*batch_size:(i+1)*batch_size], y_train_onehot[i*batch_size:(i+1)*batch_size]
            _, cost_val = sess.run([train, cost], feed_dict={X: batch_xs, Y: batch_ys})
            avg_cost += cost_val / num_iterations

        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("Learning Finished!")

    # Test model and check accuracy
    print(
        "Accuracy:",
        sess.run(accuracy, feed_dict={X: x_test, Y: y_test_onehot}),
    )
    print('<------------------------------------------------------->')
    
    # Get one and predict
    r = random.randint( 0, len(x_test)-1 )

    print("Label: ", sess.run(tf.argmax(y_test_onehot[r : r + 1], axis=1)))
    print(
        "Prediction: ",
        sess.run(
            tf.argmax(hypothesis, axis=1), feed_dict={X: x_test[r : r + 1]}
        ),
    )
    
    plt.imshow(
        x_test[r : r + 1].reshape(28, 28),
        cmap="Greys",
        interpolation="nearest",
    )
    plt.show()