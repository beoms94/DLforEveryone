# Fancy Softmax Classification
import tensorflow as tf 
import numpy as np 

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed( 777 )

readData = np.loadtxt( 'data-04-zoo.csv',
                       delimiter=',', 
                       dtype = np.float32 )
x_data = readData[:, 0:-1]
y_data = readData[:, [-1]]

nb_classes = 7

# Placeholder / one_hot & Variables
X = tf.compat.v1.placeholder( tf.float32, shape=[None, 16] )
Y = tf.compat.v1.placeholder( tf.int32, shape=[None, 1] )

Y_one_hot = tf.one_hot( Y, nb_classes )
Y_one_hot = tf.reshape( Y_one_hot, [-1, nb_classes] )

W = tf.Variable( tf.compat.v1.random_normal([16, nb_classes]), name='weight' )
b = tf.Variable( tf.compat.v1.random_normal([nb_classes]), name='bias' )

# Logits & Hypothesis
logits = tf.matmul(X, W) + b
hypothesis = tf.compat.v1.nn.softmax( logits )

# Cost / loss function
cost = tf.reduce_mean( tf.compat.v1.nn.softmax_cross_entropy_with_logits_v2( logits=logits,
                                                                     labels=tf.stop_gradient([Y_one_hot])))

optimizer = tf.compat.v1.train.GradientDescentOptimizer( learning_rate=0.1).minimize( cost )
prediction = tf.argmax( hypothesis, 1 )
correct_prediction = tf.equal( prediction, tf.argmax(Y_one_hot, 1) )
accuracy = tf.reduce_mean( tf.compat.v1.cast(correct_prediction, tf.float32) )

# Launch the graph
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    for step in range(2001):
        _, cost_val, acc_val = sess.run( [optimizer, cost, accuracy],
                                         feed_dict={X:x_data, Y:y_data} )
    
        if step%100 == 0:
            print("Step: {:5}\tCost: {:.3f}\tAcc: {:.2%}".format(step, cost_val, acc_val))

    # Let's see if wee can predict
    pred = sess.run( prediction, feed_dict={X:x_data} )

    for p, y in zip(pred, y_data.flatten()):
        print("[{}] Prediction: {} True Y: {}".format(p == int(y), p, int(y)))