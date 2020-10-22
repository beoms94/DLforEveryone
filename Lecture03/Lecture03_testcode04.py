
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()

X = [1, 2, 3]
Y = [1, 2, 3]

# Set wrong model weights
W = tf.Variable( 5. )

# Linear model
hypothesis = W * X

# Manual gradient
gradient = tf.reduce_mean( (W*X - Y)*X ) * 2

# Cost / loss function
cost = tf.reduce_mean( tf.square( hypothesis - Y ) )
optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate=0.01 )

# Get gradients
gvs = optimizer.compute_gradients( cost )
# Apply gradients
apply_gradients = optimizer.apply_gradients( gvs )

# Launch the graph in a session
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(100):
        print(step, sess.run([gradient, W, gvs]) )
        sess.run( apply_gradients )