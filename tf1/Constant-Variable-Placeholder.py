# Tensorflow day 1
import tensorflow as tf
import os

# Ignore the Warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# For Constant
A = tf.constant('Hello Tensorflow!')
B = tf.constant(10, dtype = tf.int64, shape = [3,4], name = 'Messi')
with tf.Session() as sess:
    print('************')
    print(A)
    print('************')
    print(sess.run(A))
    print('************')
    print(B)
    print('************')
    print(sess.run(B))

# For Variable
C = tf.Variable(0, dtype = tf.int64)
sess = tf.Session()
# Initialize Variable
sess.run(tf.global_variables_initializer())
for i in range(5):
    print(sess.run(C.assign(i)))

# For Placeholder
D = tf.placeholder(dtype = tf.int64)
E = tf.placeholder(dtype = tf.int64)
F = tf.placeholder(dtype = tf.int64)
F = D + E
result = sess.run(F, feed_dict={D:11,E:99})
print('************')
print(result)


