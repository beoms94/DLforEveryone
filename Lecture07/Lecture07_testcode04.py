# Learning rate and Evaluation
import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt 
import random
import math

from PIL import Image

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

# Load MNIST data set
nb_classes = 10
mnist = tf.keras.datasets.mnist 
mnist = mnist.load_data()

num_example = len( mnist[0][0] )

mnist_train_images = mnist[0][0]
mnist_train_images = mnist_train_images.reshape( num_example, 28*28 )
mnist_test_images = mnist[1][0]
mnist_test_images = mnist_test_images.reshape( len(mnist[1][0]), 28*28 )

for i in range( num_example ):
    mnist_train_images[i] = ( mnist_train_images[i] - np.mean(mnist_train_images[i]) ) / np.std(mnist_train_images[i])
    mnist_train_images[i] = np.float32( mnist_train_images[i] )
    
    if i%600 == 0:
        print('---Standardizing '+ str( float(i)*100/float(num_example) ) +' percent finished---' )


print("\n <------------ Standardization finished ------------> \n")

mnist_train_labels = mnist[0][1]
mnist_train_labels = mnist_train_labels.reshape( num_example, 1 )
mnist_onehot = np.zeros( (num_example, 10), dtype=np.float32 )
for k in range(num_example):
    temp_num = mnist_train_labels[k]
    mnist_onehot[k][temp_num] = 1.0

# Placeholder & Variables
X = tf.compat.v1.placeholder(tf.float32, shape=[None, 784] )
Y = tf.compat.v1.placeholder(tf.float32, shape=[None, nb_classes] )
W = tf.Variable( tf.compat.v1.random_normal([784, nb_classes]) )
b = tf.Variable( tf.compat.v1.random_normal([nb_classes]))

# Hypothesis & cost function & optimizer
hypothesis = tf.compat.v1.nn.softmax( tf.matmul(X, W) + b )

cost = tf.reduce_mean( -tf.reduce_sum(Y*tf.compat.v1.log(hypothesis), axis=1) )
train = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize( cost )

# Test model & Calculate accuracy
is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# Parameters
num_epochs = 15
batch_size = 100
num_iterations = int( num_example/batch_size )

# Launch the graph
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    # Training cycle
    for epoch in range(num_epochs):
        avg_cost = 0

        for i in range(num_iterations):
            batch_xs, batch_ys = mnist_train_images[100*i : 100*(i+1)], mnist_onehot[100*i : 100*(i+1)]
            _, cost_val = sess.run( [train, cost], feed_dict={X:batch_xs, Y:batch_ys} )
            avg_cost += cost_val / num_iterations
        print("Epoch: {:04d}, Cost: {:.9f}".format(epoch + 1, avg_cost))

    print("\n <------------ Learning finished ------------> \n")


'''
mnist_1 = mnist[0][0][0]

img = Image.fromarray(mnist_1, mode="L")
showing = plt.imshow( img, cmap="binary" )
plt.show(showing)

mnist_1 = mnist[0][0][0]
bb = mnist_1.reshape(28*28, 1)
print( bb )

img = Image.fromarray(mnist_train_images[0].reshape(28, 28), mode="L")
figure_1 = plt.imshow( img, cmap="binary")
plt.show(figure_1)
'''