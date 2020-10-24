# Multi Variable linear regression
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

x1_data = [73., 93., 89., 96., 73.]
x2_data = [80., 88., 91., 98., 66.]
x3_data = [75., 93., 90., 100., 70.]

y_data = [152., 185., 180., 196., 142.]

# placenoulders for a tensor that will bw always fed.
X1 = tf.compat.v1.placeholder( tf.float32 )
X2 = tf.compat.v1.placeholder( tf.float32 )
X3 = tf.compat.v1.placeholder( tf.float32 )
Y  = tf.compat.v1.placeholder( tf.float32 )

w1 = tf.Variable( tf.compat.v1.random_normal([1]), name='Weight1' )
w2 = tf.Variable( tf.compat.v1.random_normal([1]), name='Weight2' )
w3 = tf.Variable( tf.compat.v1.random_normal([1]), name='Weight3' )
b  = tf.Variable( tf.compat.v1.random_normal([1]), name='bias' )

# Hypothesis
hypothesis = X1*w1 + X2*w2 + X3*w3 + b

# Cost / loss function 
cost = tf.reduce_mean( tf.square( hypothesis - Y ) )

# Minimize. Need a very small learing rate for this data set
optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate = 1e-5 )
train = optimizer.minimize( cost )

# Launch the graph in session
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range (2001):
        cost_val, hy_val, _ = sess.run( [cost, hypothesis, train],
                                        feed_dict={X1: x1_data, X2: x2_data, X3: x3_data, Y: y_data})
        
        if step % 10 == 0:
            print( step, "Cost : ", cost_val, "\nPrediction:\n", hy_val )