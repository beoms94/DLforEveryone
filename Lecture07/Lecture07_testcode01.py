# Learning rate and Evaluation
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed( 777 )

x_data = [[1, 2, 1],
          [1, 3, 2],
          [1, 3, 4],
          [1, 5, 5],
          [1, 7, 5],
          [1, 2, 5],
          [1, 6, 6],
          [1, 7, 7]]
y_data = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1],
          [0, 1, 0],
          [0, 1, 0],
          [0, 1, 0],
          [1, 0, 0],
          [1, 0, 0]]

# Evaluation our model using this test dataset
x_test = [[2, 1, 1],
          [3, 1, 2],
          [3, 3, 4]]
y_test = [[0, 0, 1],
          [0, 0, 1],
          [0, 0, 1]]

# Placeholders and Variables
X = tf.compat.v1.placeholder( "float", shape=[None, 3] )
Y = tf.compat.v1.placeholder( "float", shape=[None, 3] )
W = tf.Variable( tf.compat.v1.random_normal([3, 3]), name='weight' )
b = tf.Variable( tf.compat.v1.random_normal([3]), name='bias' )

# Hypothesis & Cost / loss function
hypothesis = tf.compat.v1.nn.softmax( tf.matmul(X, W) + b ) 

cost = tf.reduce_mean( -tf.reduce_sum(Y*tf.compat.v1.log(hypothesis), axis = 1) )
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1).minimize( cost )

# Correct predicttion Test Model
prediction = tf.argmax( hypothesis, 1 )
is_correct = tf.equal( prediction, tf.argmax(Y, 1) )
accuracy = tf.reduce_mean( tf.compat.v1.cast(is_correct, tf.float32) )

# Launch the graph
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(201):
        cost_val, W_val, _ = sess.run( [cost, W, optimizer],
                                        feed_dict={X:x_data, Y:y_data} )

    # predict
    print("Prediction:", sess.run(prediction, feed_dict={X: x_test}))
    # Calculate the accuracy
    print("Accuracy: ", sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))