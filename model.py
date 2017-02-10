import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from gumbel_softmax import gumbel_softmax
from constants import *
import numpy as np

initial_c = tf.placeholder(tf.float32, shape=(None, HIDDEN_STATE_SIZE))
initial_h = tf.placeholder(tf.float32, shape=(None, HIDDEN_STATE_SIZE))

inputs = tf.placeholder(tf.float32, [None, SEQ_LENGTH, VOCAB_SIZE])


def generator(initial_c, initial_h):
    with tf.variable_scope("generator") as scope:
        softmax_w = tf.get_variable("softmax_w", [HIDDEN_STATE_SIZE, VOCAB_SIZE])
        softmax_b = tf.get_variable("softmax_b", [VOCAB_SIZE])

        def loop_function(prev_output, _):
            prev_output = tf.matmul(prev_output, softmax_w) + softmax_b
            return gumbel_softmax(prev_output, temperature=0.2, hard=True)

        # Create a dummy first input
        first_input = np.zeros((BATCH_SIZE, VOCAB_SIZE))
        first_input[:, 0] = 1
        first_input = tf.constant(first_input, dtype=tf.float32)

        cell = LSTMCell(HIDDEN_STATE_SIZE, state_is_tuple=True)
        outputs, states = legacy_seq2seq.rnn_decoder(decoder_inputs=20 * [first_input],
                                              initial_state=(initial_c, initial_h),
                                              cell=cell, loop_function=loop_function, scope=scope)
        logits = [tf.matmul(output, softmax_w) + softmax_b for output in outputs]
        ys = [gumbel_softmax(logit, temperature=0.2, hard=True) for logit in logits]
        ys = tf.stack(ys, axis=1)
        return ys


def discriminator(x, reuse=False):
    with tf.variable_scope("discriminator") as scope:
        if reuse:
            scope.reuse_variables()
        lstm = LSTMCell(HIDDEN_STATE_SIZE, state_is_tuple=True)
        softmax_w = tf.get_variable("softmax_w", [HIDDEN_STATE_SIZE, N_CLASSES])
        softmax_b = tf.get_variable("softmax_b", [N_CLASSES])
        lstm_outputs, _states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32, scope=scope)
        return tf.matmul(lstm_outputs[-1], softmax_w) + softmax_b


# Evaluate the losses
d_logits = discriminator(inputs)
g = generator(initial_c, initial_h)
d_logits_ = discriminator(g, reuse=True)


d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits, labels=tf.ones_like(d_logits)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_, labels=tf.zeros_like(d_logits_)))
d_loss = d_loss_real + d_loss_fake

d_loss_real_sum = tf.summary.scalar("d_loss_real", d_loss_real)
d_loss_fake_sum = tf.summary.scalar("d_loss_fake", d_loss_fake)

g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits=d_logits_, labels=tf.ones_like(d_logits_)))

g_loss_sum = tf.summary.scalar("g_loss", g_loss)
d_loss_sum = tf.summary.scalar("d_loss", d_loss)

# Optimizers
t_vars = tf.trainable_variables()
print [var.name for var in t_vars]
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

d_optim = tf.train.AdamOptimizer(LEARNING_RATE).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(LEARNING_RATE).minimize(g_loss, var_list=g_vars)

# TODO: Set up the adversarial training process
with tf.Session() as sess:
    tf.initialize_all_variables().run()
