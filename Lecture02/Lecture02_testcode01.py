
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

W = tf.Variable(tf.compat.v1.random_normal([1]), name='weight')
b = tf.Variable(tf.compat.v1.random_normal([1]), name='bias')

# Our hypothesis Wx + b
hypothesis = x_train*W + b

# Cost / loss function
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# Minimize
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize( cost )

# Launch the graph in a session
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())

    # Fit the line
    for step in range(2001):
        sess.run( train )
        if step % 20 == 0:
            print( step, sess.run(cost), sess.run(W), sess.run(b) )