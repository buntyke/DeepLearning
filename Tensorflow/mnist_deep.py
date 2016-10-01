# import modules
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load the input data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# create a session
sess = tf.InteractiveSession()

# train convolutional neural network model

# functions to initialize weight parameters
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# functions to perform convolutions and max pooling
def conv_2d(x,W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

# reshape image variable
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1,28,28,1])

# variable for true labels
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# create the convolutional layers
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

# apply convolutional layers
h_conv1 = tf.nn.relu(conv_2d(x_image, W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

h_conv2 = tf.nn.relu(conv_2d(h_pool1, W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# create a fully connected layer
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# apply fully connected layer
h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# apply dropout before computing softmax probabilities
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob=keep_prob)

# obtain output
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2)+b_fc2

# evaluate loss function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y_conv, y_))

# create optimizer
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# formulation for computing accuracy
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize all variables for the session
sess.run(tf.initialize_all_variables())

# train the model
for i in range(20000):
    batch = mnist.train.next_batch(50)
   
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print "Step %d: Accuracy %g"%(i,train_accuracy)
        
    train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
    
# obtain the test accuracy
print "Test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images, 
                                                    y_: mnist.test.labels, 
                                                    keep_prob: 1.0})
