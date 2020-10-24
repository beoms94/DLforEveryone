# Multi Variable linear regression using Matrix
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

x_data = [[73., 80., 75.],
          [93., 88., 93.],
          [89., 91., 90.],
          [96., 98., 100.],
          [73., 66., 70.]]
y_data = [[152.],
          [185.],
          [180.],
          [196.],
          [142.]]

# placeholders for a tensor that will be always fed.
X = tf.compat.v1.placeholder( tf.float32, shape=[None, 3] )
Y = tf.compat.v1.placeholder( tf.float32, shape=[None, 1] )

W = tf.Variable( tf.compat.v1.random_normal([3, 1]), name='weight' )
b = tf.Variable( tf.compat.v1.random_normal([1]), name='bias')

# Hypothesis
hypothesis = tf.matmul(X, W) + b 

# Simplified cost / loss function
cost = tf.reduce_mean( tf.square( hypothesis - Y ) )

# Minimize
optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate = 1e-5 )
train = optimizer.minimize( cost )

# Launch the graph in a session
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range (2001):
        cost_val, hy_val, _ = sess.run( [cost, hypothesis, train],
                                        feed_dict = {X: x_data, Y: y_data} )
    
        if step % 10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
