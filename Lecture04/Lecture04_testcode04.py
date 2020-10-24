
import tensorflow as tf 

tf.compat.v1.disable_eager_execution()
tf.compat.v1.set_random_seed(777)

readFile = tf.compat.v1.train.string_input_producer( ['data-01-test-score.csv'], 
                                                     shuffle = False, 
                                                     name = 'readFile' )
reader = tf.compat.v1.TextLineReader()
key, value = reader.read( readFile )

# Default values, in case of empty columns. Also specifies the type of the decoded result.
record_defaults = [ [0.], [0.], [0.], [0.] ]
readData = tf.compat.v1.decode_csv( value, 
                                    record_defaults=record_defaults )

# Collect batches of csv in
train_x_batch, train_y_batch = \
tf.compat.v1.train.batch( [readData[0:-1], readData[-1:]], batch_size = 10 )

# placeholder & Variables
X = tf.compat.v1.placeholder( tf.float32, shape = [None, 3] )
Y = tf.compat.v1.placeholder( tf.float32, shape = [None, 1] )
W = tf.Variable( tf.compat.v1.random_normal([3, 1]), name='weight' )
b = tf.Variable( tf.compat.v1.random_normal([1]), name='bias')

# Hypothesis & cost / loss function 
hypo = tf.matmul(X, W) + b 
cost = tf.reduce_mean( tf.square( hypo - Y ) )

# Optimizer 
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 1e-5 )
train = optimizer.minimize( cost )

# Launch the graph in a session
with tf.compat.v1.Session() as sess:
    sess.run( tf.compat.v1.global_variables_initializer() )

    # Start populating the readFile.
    coord = tf.compat.v1.train.Coordinator()
    threads = tf.compat.v1.train.start_queue_runners( sess=sess, coord=coord )

    for step in range(2001):
        x_batch, y_batch = sess.run( [train_x_batch, train_y_batch] )
        cost_val, hy_val, _ = sess.run( [cost, hypo, train],
                                        feed_dict={X: x_batch, Y: y_batch} )
    
        if step%10 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
    
    '''
    # Ask my score
    print("Your score will be ", sess.run(hypo, feed_dict={X: [[100, 70, 101]]}))

    print("Other scores will be ", sess.run(hypo, feed_dict={X: [[60, 70, 110], [90, 100, 80]]}))
    '''

    coord.request_stop()
    coord.join( threads )