
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print("node1:", node1, "node2:", node2)
print("node3:", node3)

with tf.compat.v1.Session() as sess:

	print("sess.run(node1, node2):",sess.run([node1, node2]))
	print("sess.run(node3):",sess.run(node3))
