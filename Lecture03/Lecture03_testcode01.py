
import tensorflow as tf
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()

X = [1, 2, 3]
Y = [1, 2, 3]

W = tf.compat.v1.placeholder( tf.float32 )
hypothesis = X*W

cost = tf.reduce_mean( tf.square( hypothesis - Y ) )

with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    W_val = []
    cost_val = []

    for i in range (-30, 50):
        feed_W = i*0.1
        curr_cost, curr_W = sess.run( [cost, W], feed_dict={W: feed_W} )
        W_val.append( curr_W )
        cost_val.append( curr_cost )

    plt.plot( W_val, cost_val )
    plt.show()