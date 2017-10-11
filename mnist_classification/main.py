#[Long Short Term Memory](http: // deeplearning.cs.cmu.edu / pdfs / Hochreiter97_lstm.pdf)
#[MNIST Dataset](http: // yann.lecun.com / exdb / mnist /).


import tensorflow as tf
from tensorflow.contrib import rnn


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/", one_hot=True)


learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200

# Network Parameters
num_input = 28 # MNIST data input (img shape: 28*28)
timesteps = 28 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x_ = tf.placeholder("float", [None, timesteps, num_input])
y_ = tf.placeholder("float", [None, num_classes])

weights = {'out': tf.Variable(tf.random_normal([num_hidden  ,num_classes]))}
biases = {'out' : tf.Variable(tf.random_normal([num_classes]))}




"""
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
"""

def RNN(x_ , weights  , biases):
    x = tf.unstack(x_, timesteps, 1)
    lstm_cell = rnn.BasicLSTMCell(num_hidden , forgot_bias=1.0)
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
logits = RNN(x_, weights, biases)
prediction = tf.nn.softmax(logits)

loss_op=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits , labels=y_))
optimizer=tf.train.GradientDiscentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction ,1) , tf.argmax(y_ , 1 ))
accuracy =tf.reduce_mean(tf.cast(correct_pred , tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)


    for step in range(1 , training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={x_: batch_x, y_: batch_y})

        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={x_: batch_x,
                                                             y_: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
              "{:.4f}".format(loss) + ", Training Accuracy= " + \
              "{:.3f}".format(acc))


    print("Optimization Finished!")


    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x_: test_data, y_: test_label}))


