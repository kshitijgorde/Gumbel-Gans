#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from gumbel_softmax import gumbel_softmax
from constants import *
import numpy as np

from data.ptb import CharLevelPTB
from data.rnnpg import CharLevelRNNPG
from data.oracle import OracleDataloader, OracleVerifier
from utils import sample_Z

SEQ_LENGTH, VOCAB_SIZE, TEST_SIZE, c = None, None, None, None

if DATASET == 'ptb':
    c = CharLevelPTB()
    VOCAB_SIZE = PTB_VOCAB_SIZE
    SEQ_LENGTH = PTB_SEQ_LENGTH
elif DATASET == 'pg':
    c = CharLevelRNNPG()
    VOCAB_SIZE = PG_VOCAB_SIZE
    SEQ_LENGTH = PG_SEQ_LENGTH
elif DATASET == 'oracle':
    c = OracleDataloader(BATCH_SIZE, ORACLE_VOCAB_SIZE)
    VOCAB_SIZE = ORACLE_VOCAB_SIZE
    SEQ_LENGTH = ORACLE_SEQ_LENGTH
    TEST_SIZE = ORACLE_TEST_SIZE

initial_c = tf.placeholder(tf.float32, shape=(None, HIDDEN_STATE_SIZE))
initial_h = tf.placeholder(tf.float32, shape=(None, HIDDEN_STATE_SIZE))

inputs_pre = tf.placeholder(tf.float32, [None, SEQ_LENGTH-1, VOCAB_SIZE])
inputs = tf.placeholder(tf.float32, [None, SEQ_LENGTH-1, VOCAB_SIZE])

targets = tf.placeholder(tf.float32, [None, SEQ_LENGTH-1, VOCAB_SIZE])


def generator(initial_c, initial_h, mode='gan', inputs=None, targets=None, reuse=False):
    assert mode in ['test', 'pretrain', 'gan']
    with tf.variable_scope("generator") as scope:
        if reuse:
            scope.reuse_variables()
        softmax_w = tf.get_variable("softmax_w", [HIDDEN_STATE_SIZE, VOCAB_SIZE])
        softmax_b = tf.get_variable("softmax_b", [VOCAB_SIZE])

        def loop_function(prev_output, _):
            prev_output = tf.matmul(prev_output, softmax_w) + softmax_b
            return gumbel_softmax(prev_output, temperature=0.2, hard=True)

        # Create a dummy first input
        first_input = np.zeros((BATCH_SIZE, VOCAB_SIZE))
        first_input[:, c.start_char_idx] = 1
        first_input = tf.constant(first_input, dtype=tf.float32)

        cell = LSTMCell(HIDDEN_STATE_SIZE, state_is_tuple=True)
        if mode == 'pretrain':
            inputs = tf.unstack(inputs, axis=1)
            outputs, states = legacy_seq2seq.rnn_decoder(decoder_inputs=inputs,
                                                         initial_state=(initial_c, initial_h),
                                                         cell=cell, loop_function=None, scope=scope)
            logits = [tf.matmul(output, softmax_w) + softmax_b for output in outputs]

            targets = tf.unstack(tf.transpose(targets, [1, 0, 2]))

            loss = [tf.nn.softmax_cross_entropy_with_logits(labels=target, logits=logit)
                    for target, logit in zip(targets, logits)] # ignore start token
            loss = tf.reduce_mean(loss)
            return loss
        elif mode == 'test':
            outputs, states = legacy_seq2seq.rnn_decoder(decoder_inputs=(SEQ_LENGTH-1) * [first_input],
                                                         initial_state=(initial_c, initial_h),
                                                         cell=cell, loop_function=loop_function, scope=scope)
            logits = [tf.matmul(output, softmax_w) + softmax_b for output in outputs]
            return tf.transpose(tf.argmax(logits, axis=2))
        else:
            outputs, states = legacy_seq2seq.rnn_decoder(decoder_inputs=(SEQ_LENGTH-1) * [first_input],
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
        lstm = LSTMCell(HIDDEN_STATE_SIZE_D, state_is_tuple=True)
        softmax_w = tf.get_variable("softmax_w", [HIDDEN_STATE_SIZE_D, N_CLASSES])
        softmax_b = tf.get_variable("softmax_b", [N_CLASSES])
        lstm_outputs, _states = tf.nn.dynamic_rnn(lstm, x, dtype=tf.float32, scope=scope)
        return tf.matmul(lstm_outputs[-1], softmax_w) + softmax_b

def generate_test_file(g_test, sess, eval_file):
    z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
    c_z, h_z = np.vsplit(z, 2)

    with open(eval_file, 'w') as f:
        for _ in xrange(TEST_SIZE/BATCH_SIZE):
            oracle_out = sess.run([g_test], feed_dict={
                initial_c: c_z,
                initial_h: h_z
            })
            for l in oracle_out[0]:
                buffer = ' '.join([str(x) for x in l]) + '\n'
                f.write(buffer)

# Evaluate the losses
d_logits = discriminator(inputs)
g_pre_loss = generator(initial_c, initial_h, mode='pretrain', inputs=inputs_pre, targets=targets)

# Optimizers
t_vars = tf.trainable_variables()
print [var.name for var in t_vars]
g_vars = [var for var in t_vars if var.name.startswith('generator')]
d_vars = [var for var in t_vars if var.name.startswith('discriminator')]

g_pre_optim = tf.train.AdamOptimizer(LEARNING_RATE_PRE_G, name="Adam_g_pre").minimize(g_pre_loss, var_list=g_vars)
g_pre_loss_sum = tf.summary.scalar("g_pre_loss", g_pre_loss)

g = generator(initial_c, initial_h, reuse=True)
g_test = generator(initial_c, initial_h, reuse=True, mode='test')
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

d_optim = tf.train.GradientDescentOptimizer(LEARNING_RATE_D).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(LEARNING_RATE_G, name="Adam_g").minimize(g_loss, var_list=g_vars)

with tf.Session() as sess:
    if DATASET == 'oracle':
        t=OracleVerifier(BATCH_SIZE, sess)
    tf.initialize_all_variables().run()
    writer = tf.summary.FileWriter("./logs", sess.graph)
    counter = 1

    if DATASET == 'oracle':
        generate_test_file(g_test, sess, t.eval_file)
        print "NLL Oracle Loss before training: %.8f" % t.get_loss()

    for pre_epoch in xrange(PRETRAIN_EPOCHS):
        batch_idx = 1
        for batch in c.get_train_batch(BATCH_SIZE):
            batch = c.convert_batch_to_input_target(batch)
            batch_input, batch_targets = batch

            z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
            c_z, h_z = np.vsplit(z, 2)
            _, g_pre_loss_curr, summary_str = sess.run([g_pre_optim, g_pre_loss, g_pre_loss_sum], feed_dict={
                inputs_pre: batch_input,
                initial_c: c_z,
                initial_h: h_z,
                targets: batch_targets
            })
            writer.add_summary(summary_str, counter)
            batch_idx += 1
            print "Epoch: [%d] Batch: %d, g_pre_loss: %.8f" % (pre_epoch, batch_idx, g_pre_loss_curr)
        if DATASET == 'oracle':
            generate_test_file(g_test, sess, t.eval_file)
            print "NLL Oracle Loss after pre-train epoch %d: %.8f" % (pre_epoch, t.get_loss())

    counter = 1
    for epoch in xrange(N_EPOCHS):
        batch_idx = 0
        for batch in c.get_train_batch(BATCH_SIZE):

            # To remove start token, since generator doesn't generate it either
            batch = c.convert_batch_to_input_target(batch)
            _, batch_targets = batch

            z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
            c_z, h_z = np.vsplit(z, 2)
            _, d_loss_curr, summary_str = sess.run([d_optim, d_loss, d_loss_sum], feed_dict={
                inputs: batch_targets,
                initial_c: c_z,
                initial_h: h_z
            })
            writer.add_summary(summary_str, counter)

            _, g_loss_curr, summary_str = sess.run([g_optim, g_loss, g_loss_sum], feed_dict={
                initial_c: c_z,
                initial_h: h_z
            })
            # writer.add_summary(summary_str, counter)

            _, g_loss_curr, summary_str = sess.run([g_optim, g_loss, g_loss_sum], feed_dict={
                initial_c: c_z,
                initial_h: h_z
            })
            writer.add_summary(summary_str, counter)

            batch_idx += 1
            counter += 1
            print "Epoch: [%d] Batch: %d d_loss: %.8f, g_loss: %.8f" % (epoch, batch_idx, d_loss_curr, g_loss_curr)

        if DATASET == 'oracle':
            generate_test_file(g_test, sess, t.eval_file)
            print "NLL Oracle Loss after epoch %d: %.8f" % (epoch, t.get_loss())

    # TODO: sample generated text



