
import tensorflow as tf 
import numpy as np 

tf.compat.v1.disable_eager_execution() 
tf.compat.v1.set_random_seed(777)

readData = np.loadtxt( 'data-01-test-score.csv',
                       delimiter = ',',
                       dtype = np.float32 )
x_data = readData[ :, 0:-1 ]
y_data = readData[ :, [-1] ]

print( "\nx_data shape:", x_data.shape )
print( "\ny_data shape:", y_data.shape )

# placeholder for a tensor that will be always fed.
X = tf.compat.v1.placeholder( tf.float32, shape=[None, 3] )
Y = tf.compat.v1.placeholder( tf.float32, shape=[None, 1] )

W = tf.Variable( tf.compat.v1.random_normal([3, 1]), name='weight' )
b = tf.Variable( tf.compat.v1.random_normal([1]), name='bias' )

# Hypothesis
hypothesis = tf.matmul(X, W) + b

# Simplified cost / loss function
cost = tf.reduce_mean( tf.square( hypothesis - Y ) )

# Optimizer 
optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate=1e-5 )
train = optimizer.minimize( cost )

# Launch the graph in a session
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(2001):
        cost_val, hy_val, _ = sess.run( [cost, hypothesis, train],
                                        feed_dict={X: x_data, Y: y_data} )
        
        if step % 10 == 0:
            print(step, "Cost:", cost_val, "\nPrediction:\n", hy_val)
