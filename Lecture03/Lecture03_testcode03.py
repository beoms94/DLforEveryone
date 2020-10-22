
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()

X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable(5.0)

# Linear model
hypothesis = X*W 

# Cost / loss function 
cost = tf.reduce_mean( tf.square( hypothesis - Y ) )

# Minimize: Gradient Descent Magic
optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate = 0.1 )
train = optimizer.minimize( cost )

# Launch the graph in a session
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(100):
        print(step, sess.run(W) )
        sess.run( train )
