'''
    Y = softmax(X*W + b)
    Trainning = computing variables W and b
'''
import tensorflow as tf
import tensorflowvisu
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

print("Tensorflow version " + tf.__version__)

tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets("data", one_hot=True, reshape=False, validation_size=0)

# None will be the number of images in this trainning batch
# 1 the number of values per pixel
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
#Y_ = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

init = tf.initialize_all_variables()

# model
# pass -1 to flatting images
Y = tf.nn.softmax(tf.matmul(tf.reshape(X, [-1, 784]), W) + b)
# placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, 10])

# loss function
cross_entropy = -tf.reduct_sum(Y_ * tf.log(Y))

# % for correct answers fund in batch
is_correct = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)   # 0.003 is learning rate
# cross_entropy is loss function
train_step = optimizer.minimize(cross_entropy)


sess = tf.Session()
sess.run(init)

for i in range(1000):
    batch_X, batch_Y = mnist.train.next_batch(100)
    train_data = {X: batch_X, Y:batch_Y}

    # train
    sess.run(train_step, feed_dict = train_data)
    
    # success ?
    a, c = sess.run([accuracy, cross_entropy], feed_dict=train_data)

    # success on test data?
    test_data = {X: mnist.test.images, Y_:mnist.test.labels}
    a, c = sess.run([accuracy, cross_entropy], feed= test_data)

    
