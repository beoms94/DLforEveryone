# Logistic Regression Classifier
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

x_data = [[1, 2], 
          [2, 3],
          [3, 1],
          [4, 3],
          [5, 3],
          [6, 2]]
y_data = [[0], 
          [0],
          [0],
          [1],
          [1],
          [1]]

X = tf.compat.v1.placeholder( tf.float32, shape=[None, 2] )
Y = tf.compat.v1.placeholder( tf.float32, shape=[None, 1] )

W = tf.Variable( tf.compat.v1.random_normal([2, 1]), name='Weight' )
b = tf.Variable( tf.compat.v1.random_normal([1]), name='bias' )

# Hypothesis using sigmoid 
hypothesis = tf.sigmoid( tf.matmul(X, W) + b )

# Cost / loss function
cost = -tf.reduce_mean( Y*tf.compat.v1.log(hypothesis) 
                       + (1-Y)*tf.compat.v1.log(1-hypothesis) )

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize( cost )

# Accuracy computation
# True if hypothesis > 0.5, else False
predicted = tf.compat.v1.cast( hypothesis > 0.5, dtype=tf.float32 )
accuracy = tf.reduce_mean( tf.compat.v1.cast(tf.equal(predicted, Y), dtype=tf.float32) )

# Luanch graph
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(10001):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data} )

        if step%200 == 0:
            print( step, cost_val )

    # Accuracy report
    h, c, a = sess.run([hypothesis, predicted, accuracy],
                        feed_dict={X:x_data, Y:y_data} )

    print("\nHypothesis: ", h, "\nCorrect (Y): ", c, "\nAccuracy: ", a)