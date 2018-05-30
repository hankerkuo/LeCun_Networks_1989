import tensorflow as tf
from six.moves import cPickle as pickle
import os
from filter_operators import *


def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs})
    return result


def weight_variable(shape):
    # truncated normal distribution
    # initial = tf.truncated_normal(shape, stddev=0.1)
    if np.shape(shape) == (4,):
        initial = tf.random_uniform(shape, minval=-2.4 / (shape[0] * shape[1]), maxval=2.4 / (shape[0] * shape[1]))
    elif np.shape(shape) == (2,):
        initial = tf.random_uniform(shape, minval=-2.4 / shape[0], maxval=2.4 / shape[0])
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    # stride [Batch, Height, Width, Channel], in the computer's point if view, it sees just four dimensions
    # so we shouldn't pass through any of samples(batch) or channels
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')


def max_pool_2x2(x):
    # stride [Batch, Height, Width, Channel]
    # ksize [Batch, Height, Width, Channel]
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# define step
global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 16, 16], name='x_input')   # 16*16
    ys = tf.placeholder(tf.float32, [None, 10], name='y_input')
x_image = tf.reshape(xs, [-1, 16, 16, 1], name='reshape')

# conv1 (locally connected)
conv1_padding = tf.keras.layers.ZeroPadding2D(
    padding=1,
    data_format='channels_last')(x_image)
w_x_b_conv1 = tf.keras.layers.LocallyConnected2D(
    filters=1,
    strides=2,
    kernel_size=[3, 3],
    padding='valid',
    data_format='channels_last',)(conv1_padding)
h_conv1 = 1.7159 * tf.nn.tanh((2 / 3) * w_x_b_conv1)

# conv2 (locally connected)
conv2_padding = tf.keras.layers.ZeroPadding2D(
    padding=2,
    data_format='channels_last')(h_conv1)
w_x_b_conv2 = tf.keras.layers.LocallyConnected2D(
    filters=1,
    strides=2,
    kernel_size=[5, 5],
    padding='valid',
    data_format='channels_last',)(conv2_padding)
h_conv2 = 1.7159 * tf.nn.tanh((2 / 3) * w_x_b_conv2)

# FC layer
with tf.name_scope('FC'):
    with tf.name_scope('weight'):
        W_fc1 = weight_variable([4*4*1, 10])
    with tf.name_scope('bias'):
        b_fc1 = bias_variable([10])
    # [n_samples, 4, 4, 1] ->> [n_samples, 4*4*1]
    with tf.name_scope('flat'):
        h_pool2_flat = tf.reshape(h_conv2, [-1, 4*4*1])
    with tf.name_scope('Wx_plus_bias'):
        h_FC = tf.matmul(h_pool2_flat, W_fc1) + b_fc1
with tf.name_scope('prediction'):
    prediction = tf.nn.softmax(h_FC)
    # prediction = 1.7159 * tf.nn.tanh((2 / 3) * (tf.matmul(h_pool2_flat, W_fc1) + b_fc1))

# the error between prediction and real data, two kinds of cost function
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
with tf.name_scope('loss'):
    mse = tf.reduce_mean(0.5 * tf.square(ys - prediction))
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mse, global_step=global_step)

# define the accuracy estimating tensor
with tf.name_scope('accuracy'):
    same_or_not = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(same_or_not, tf.float32))

saver = tf.train.Saver()
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoint/net3/save_net.ckpt'))
if ckpt and ckpt.model_checkpoint_path:
     saver.restore(sess, ckpt.model_checkpoint_path)
# saver.restore(sess, "./save_net.ckpt")

with open('./train_data/data.pickle', 'rb') as f:
    tr_dat = pickle.load(f)
with open('./train_data/label.pickle', 'rb') as f:
    tr_lab = pickle.load(f)
with open('./test_data/data.pickle', 'rb') as f:
    te_dat = pickle.load(f)
with open('./test_data/label.pickle', 'rb') as f:
    te_lab = pickle.load(f)

# summary
with tf.name_scope("summaries"):
    tf.summary.scalar("loss", mse)
    tf.summary.scalar("accuracy", accuracy)
    tf.summary.histogram("histogram loss", mse)
    summary_op = tf.summary.merge_all()

# write log files using a FileWriter
# access the tensorboard, -> tensorboard --logdir=C:\data\tensorboard\net4 , in this tf version no '' for logdir!!
writer_train = tf.summary.FileWriter('C:/data/tensorboard/net3/original/train/', sess.graph)
writer_test = tf.summary.FileWriter('C:/data/tensorboard/net3/original/test/', sess.graph)

# training process starts
batch_size = 32
for epoch in range(3000):       # epoch amount
    for batch in range(len(tr_dat) // batch_size):
        train_op, loss = sess.run([train_step, mse], feed_dict={
                                        xs: tr_dat[batch * batch_size: (batch + 1) * batch_size],
                                        ys: tr_lab[batch * batch_size: (batch + 1) * batch_size]})
        # incremental average (refresh average loss after every epoch)
        try:
            average_loss += 1 / (batch + 1) * (loss - average_loss)
        except Exception as e:
            average_loss = 0
    if (epoch + 1) % 100 == 0:
        print((epoch + 1), 'th test accuracy = %.3f' % compute_accuracy(te_dat, te_lab), end=' ')
        print('train accuracy = %.3f' % compute_accuracy(tr_dat, tr_lab), end=' ')
        print('(loss = %.4f)' % average_loss)
        summary_test = sess.run(summary_op, feed_dict={xs: te_dat, ys: te_lab})
        summary_train = sess.run(summary_op, feed_dict={xs: tr_dat, ys: tr_lab})
        # save check point (named by the number of mini batch which has already fed into the NN)
        saver.save(sess, './checkpoint/net4/save_net.ckpt', global_step=(epoch + 1) * (len(tr_dat) // batch_size))
        writer_test.add_summary(summary_test, global_step=(epoch + 1) * (len(tr_dat) // batch_size))
        writer_train.add_summary(summary_train, global_step=(epoch + 1) * (len(tr_dat) // batch_size))
    average_loss = 0

writer_test.close()
writer_train.close()
sess.close()
