
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

# See http://stackoverflow.com/question/36693740/
# X and Y data
X = tf.compat.v1.placeholder( tf.float32 )
Y = tf.compat.v1.placeholder( tf.float32 )

# Weight and bias 
W = tf.Variable( tf.compat.v1.random_normal([1]), name='Weight' )
b = tf.Variable( tf.compat.v1.random_normal([1]), name='bias' )

# hypothesis Wx + b
hypothesis = X*W + b

# Cost / loss function
cost = tf.reduce_mean( tf.square( hypothesis - Y ) )

# Minimize
optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate = 0.01 )
train = optimizer.minimize( cost )

# Launch the graph in a session
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    # Fit the line
    for step in range(2001):
        cost_val, W_val, b_val, _ = \
        sess.run( [cost, W, b, train], feed_dict={ X: [1, 2, 3], Y: [1, 2, 3] } )
        
        if step%20 == 0:
            print( step, cost_val, W_val, b_val )