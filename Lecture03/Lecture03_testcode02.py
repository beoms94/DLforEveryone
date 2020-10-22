
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()

X_data = [1, 2, 3]
Y_data = [1, 2, 3]

W = tf.Variable( tf.compat.v1.random_normal([1]), name='Weight' )
X = tf.compat.v1.placeholder( tf.float32 )
Y = tf.compat.v1.placeholder( tf.float32 )

# Our hypothesis for linear model
hypothesis = X*W

# Cost / loss function
cost = tf.reduce_sum( tf.square( hypothesis - Y ) )

# Minimize: Gradient Descent using derivative: W -= learning_rate*derivative
learning_rate = 0.1
gradient = tf.reduce_mean( (W*X - Y)*X )
descent = W - learning_rate * gradient
update = W.assign( descent )

# Launch the graph in a session.
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(21):
        sess.run( update, feed_dict={X: X_data, Y: Y_data} )
        print( step, sess.run( cost, feed_dict={X: X_data, Y: Y_data}), sess.run(W) )