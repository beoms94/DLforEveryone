# Softmax Regression
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed( 777 )

x_data = [[1, 2, 1, 1],
          [2, 1, 3, 2],
          [3, 1, 3, 4],
          [4, 1, 5, 5],
          [1, 7, 5, 5],
          [1, 2, 5, 6],
          [1, 6, 6, 6],
          [1, 7, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Placeholder & Variables
X = tf.compat.v1.placeholder( "float", shape = [None, 4])
Y = tf.compat.v1.placeholder( "float", shape = [None, 3])
nb_classes = 3

W = tf.Variable( tf.compat.v1.random_normal([4, nb_classes]), name='weight' )
b = tf.Variable( tf.compat.v1.random_normal([nb_classes]), name='bias' )

# Hypothesis
hypothesis = tf.compat.v1.nn.softmax( tf.matmul(X, W) + b )

# Cost / loss function
cost = tf.reduce_mean( -tf.reduce_sum( Y * tf.compat.v1.log(hypothesis), axis = 1) )

# Optimizer
optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate = 0.1)
train = optimizer.minimize( cost )

# Launch the graph
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(2001):
        cost_val, _ = sess.run( [cost, train], feed_dict = {X:x_data, Y:y_data} )

        if step%200 == 0:
            print( step, cost_val )

    print('--------------')
    # Testing & One-hot encoding
    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9]]})
    print(a, sess.run(tf.argmax(a, 1)))

    print('--------------')
    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4, 3]]})
    print(b, sess.run(tf.argmax(b, 1)))

    print('--------------')
    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0, 1]]})
    print(c, sess.run(tf.argmax(c, 1)))

    print('--------------')
    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7, 9], [1, 3, 4, 3], [1, 1, 0, 1]]})
    print(all, sess.run(tf.argmax(all, 1)))