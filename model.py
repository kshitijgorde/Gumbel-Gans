#!/usr/bin/env python
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq
from gumbel_softmax import gumbel_softmax
from constants import *
import numpy as np
import os

from data.ptb import CharLevelPTB
from data.rnnpg import CharLevelRNNPG
from data.oracle import OracleDataloader, OracleVerifier
from utils import *

from orthogonal import orthogonal_initializer

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='gumbel-gans')

    parser.add_argument('--dataset', type=str, default=DATASET, choices=['ptb', 'pg', 'oracle'])
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--pretrain_epochs', type=int, default=PRETRAIN_EPOCHS)
    parser.add_argument('--lr_pretrain', type=float, default=LEARNING_RATE_PRE_G)
    parser.add_argument('--hidden_size_g', type=int, default=HIDDEN_STATE_SIZE)
    parser.add_argument('--hidden_size_d', type=int, default=HIDDEN_STATE_SIZE_D)
    parser.add_argument('--lr_g', type=float, default=LEARNING_RATE_G)
    parser.add_argument('--lr_d', type=float, default=LEARNING_RATE_D)
    parser.add_argument('--epochs', type=int, default=N_EPOCHS)
    parser.add_argument('--num_d', type=int, default=NUM_D)
    parser.add_argument('--num_g', type=int, default=NUM_G)

    return parser.parse_args()

args = parse_args()

DATASET = args.dataset
BATCH_SIZE = args.batch_size
PRETRAIN_EPOCHS = args.pretrain_epochs
LEARNING_RATE_PRE_G = args.lr_pretrain
HIDDEN_STATE_SIZE = args.hidden_size_g
HIDDEN_STATE_SIZE_D = args.hidden_size_d
LEARNING_RATE_G = args.lr_g
LEARNING_RATE_D = args.lr_d
N_EPOCHS = args.epochs
NUM_D = args.num_d
NUM_G = args.num_g

from decimal import Decimal

# define other constants
LOG_LOCATION = './logs/' + DATASET[:2] + '_g' + str(NUM_G) + 'd' + str(NUM_D) + '_g' + str(HIDDEN_STATE_SIZE) + '_d' + str(HIDDEN_STATE_SIZE_D) + '_pe' + str(PRETRAIN_EPOCHS) + '_pl' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '_l'+ str("{:.0e}".format(Decimal(LEARNING_RATE_G))) + '/'
PRETRAIN_CHK_FOLDER = './checkpoints/'  +  DATASET[:2] + '_p_h' + str(HIDDEN_STATE_SIZE) + '_l' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '_e' + str(PRETRAIN_EPOCHS) + '/'
SAVE_FILE_PRETRAIN = PRETRAIN_CHK_FOLDER + DATASET[:2] + '_p_h' + str(HIDDEN_STATE_SIZE) + '_l' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '.chk'
LOAD_FILE_PRETRAIN = SAVE_FILE_PRETRAIN + 'b'
GAN_CHK_FOLDER = './checkpoints/' +  DATASET[:2] + '_g' + str(NUM_G) + 'd' + str(NUM_D) + '_g' + str(HIDDEN_STATE_SIZE) + '_d' + str(HIDDEN_STATE_SIZE_D) + '_pe' + str(PRETRAIN_EPOCHS) + '_pl' + str("{:.0e}".format(Decimal(LEARNING_RATE_PRE_G))) + '_l'+ str("{:.0e}".format(Decimal(LEARNING_RATE_G))) + '/'
SAVE_FILE_GAN = GAN_CHK_FOLDER + 'chk'
LOAD_FILE_GAN = SAVE_FILE_GAN

def print_hyperparam_consts():
    print('DATASET: ' + str(DATASET))
    print('BATCH_SIZE: ' + str(BATCH_SIZE))
    print('PRETRAIN_EPOCHS: ' + str(PRETRAIN_EPOCHS))
    print('LEARNING_RATE_PRE_G: ' + str(LEARNING_RATE_PRE_G))
    print('HIDDEN_STATE_SIZE: ' + str(HIDDEN_STATE_SIZE))
    print('HIDDEN_STATE_SIZE_D: ' + str(HIDDEN_STATE_SIZE_D))
    print('LEARNING_RATE_G: ' + str(LEARNING_RATE_G))
    print('LEARNING_RATE_D: ' + str(LEARNING_RATE_D))
    print('N_EPOCHS: ' + str(N_EPOCHS))
    print('NUM_D: ' + str(NUM_D))
    print('NUM_G: ' + str(NUM_G))
    print('LOG_LOCATION: ' + str(LOG_LOCATION))
    print('PRETRAIN_CHK_FOLDER: ' + str(PRETRAIN_CHK_FOLDER))
    print('SAVE_FILE_PRETRAIN: ' + str(SAVE_FILE_PRETRAIN))
    print('LOAD_FILE_PRETRAIN: ' + str(LOAD_FILE_PRETRAIN))
    print('GAN_CHK_FOLDER: ' + str(GAN_CHK_FOLDER))
    print('SAVE_FILE_GAN: ' + str(SAVE_FILE_GAN))
    print('LOAD_FILE_GAN: ' + str(LOAD_FILE_GAN))

print_hyperparam_consts()

# create dirs that don't exist
if SAVE_FILE_PRETRAIN:
    create_dir_if_not_exists('/'.join(SAVE_FILE_PRETRAIN.split('/')[:-1]))
if SAVE_FILE_GAN:
    create_dir_if_not_exists('/'.join(SAVE_FILE_GAN.split('/')[:-1]))
create_dir_if_not_exists(LOG_LOCATION)

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

        cell = LSTMCell(HIDDEN_STATE_SIZE, state_is_tuple=True, reuse=reuse, initializer=orthogonal_initializer())
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
        lstm = LSTMCell(HIDDEN_STATE_SIZE_D, state_is_tuple=True, reuse=reuse, initializer=orthogonal_initializer())
        softmax_w = tf.get_variable("softmax_w", [HIDDEN_STATE_SIZE_D, N_D_CLASSES])
        softmax_b = tf.get_variable("softmax_b", [N_D_CLASSES])
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

    def generate_samples():
        z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
        c_z, h_z = np.vsplit(z, 2)

        samples = sess.run(g, feed_dict={
            initial_c: c_z,
            initial_h: h_z
        })

        decoded_samples = []
        for i in xrange(len(samples)):
            decoded = []
            for j in xrange(len(samples[i])):
                decoded.append(c.idx2char[samples[i][j]])
            decoded_samples.append(tuple(decoded))
        return decoded_samples

    if DATASET == 'oracle':
        t=OracleVerifier(BATCH_SIZE, sess)
    tf.initialize_all_variables().run()
    writer = tf.summary.FileWriter(LOG_LOCATION, sess.graph)
    counter = 1

    saver = tf.train.Saver()

    if LOAD_FILE_PRETRAIN:
        # saver = tf.train.import_meta_graph(LOAD_FILE_PRETRAIN + '.meta')
        saver.restore(sess, LOAD_FILE_PRETRAIN)
        if DATASET == 'oracle':
            generate_test_file(g_test, sess, t.eval_file)
            print "NLL Oracle Loss after loading model: %.8f" % t.get_loss()
        else:
            batch_idx = 0
            ltot = 0.
            for batch in c.get_valid_batch(BATCH_SIZE):
                batch = c.convert_batch_to_input_target(batch)
                batch_input, batch_targets = batch

                z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                c_z, h_z = np.vsplit(z, 2)
                g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                    inputs_pre: batch_input,
                    initial_c: c_z,
                    initial_h: h_z,
                    targets: batch_targets
                })
                ltot += g_pre_loss_curr
                writer.add_summary(summary_str, counter)
                batch_idx += 1
            print "Valid g_pre_loss after loading model: %.8f" % (ltot/batch_idx)
            batch_idx = 0
            ltot = 0.
            for batch in c.get_test_batch(BATCH_SIZE):
                batch = c.convert_batch_to_input_target(batch)
                batch_input, batch_targets = batch

                z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                c_z, h_z = np.vsplit(z, 2)
                g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                    inputs_pre: batch_input,
                    initial_c: c_z,
                    initial_h: h_z,
                    targets: batch_targets
                })
                ltot += g_pre_loss_curr
                writer.add_summary(summary_str, counter)
                batch_idx += 1
            print "Test g_pre_loss after loading model: %.8f" % (ltot/batch_idx)
    else:
        if DATASET == 'oracle':
            generate_test_file(g_test, sess, t.eval_file)
            print "NLL Oracle Loss before training: %.8f" % t.get_loss()
        else:
            batch_idx = 0
            ltot = 0.
            for batch in c.get_valid_batch(BATCH_SIZE):
                batch = c.convert_batch_to_input_target(batch)
                batch_input, batch_targets = batch

                z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                c_z, h_z = np.vsplit(z, 2)
                g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                    inputs_pre: batch_input,
                    initial_c: c_z,
                    initial_h: h_z,
                    targets: batch_targets
                })
                ltot += g_pre_loss_curr
                writer.add_summary(summary_str, counter)
                batch_idx += 1
            print "Valid g_pre_loss before training: %.8f" % (ltot/batch_idx)
            batch_idx = 0
            ltot = 0.
            for batch in c.get_test_batch(BATCH_SIZE):
                batch = c.convert_batch_to_input_target(batch)
                batch_input, batch_targets = batch

                z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                c_z, h_z = np.vsplit(z, 2)
                g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                    inputs_pre: batch_input,
                    initial_c: c_z,
                    initial_h: h_z,
                    targets: batch_targets
                })
                ltot += g_pre_loss_curr
                writer.add_summary(summary_str, counter)
                batch_idx += 1
            print "Test g_pre_loss before training: %.8f" % (ltot/batch_idx)

        best_prtr_val_loss = float('Inf')
        for pre_epoch in xrange(PRETRAIN_EPOCHS):
            batch_idx = 0
            train_ltot = 0.
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
                train_ltot += g_pre_loss_curr
                writer.add_summary(summary_str, counter)
                batch_idx += 1
            print "Train Loss after pre-train epoch %d: %.8f" % (pre_epoch, train_ltot / batch_idx)
                # print "Epoch: [%d] Batch: %d, g_pre_loss: %.8f" % (pre_epoch, batch_idx, g_pre_loss_curr)
            if DATASET == 'oracle':
                generate_test_file(g_test, sess, t.eval_file)
                print "NLL Oracle Loss after pre-train epoch %d: %.8f" % (pre_epoch, t.get_loss())
            else:
                batch_idx = 0
                ltot = 0.
                for batch in c.get_valid_batch(BATCH_SIZE):
                    batch = c.convert_batch_to_input_target(batch)
                    batch_input, batch_targets = batch

                    z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                    c_z, h_z = np.vsplit(z, 2)
                    g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                        inputs_pre: batch_input,
                        initial_c: c_z,
                        initial_h: h_z,
                        targets: batch_targets
                    })
                    ltot += g_pre_loss_curr
                    writer.add_summary(summary_str, counter)
                    batch_idx += 1
                print "Valid Loss after pre-train epoch %d: %.8f" % (pre_epoch, ltot / batch_idx)
                batch_idx = 0
                ltot = 0.
                for batch in c.get_test_batch(BATCH_SIZE):
                    batch = c.convert_batch_to_input_target(batch)
                    batch_input, batch_targets = batch

                    z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                    c_z, h_z = np.vsplit(z, 2)
                    g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                        inputs_pre: batch_input,
                        initial_c: c_z,
                        initial_h: h_z,
                        targets: batch_targets
                    })
                    ltot += g_pre_loss_curr
                    writer.add_summary(summary_str, counter)
                    batch_idx += 1
                print "Test Loss after pre-train epoch %d: %.8f" % (pre_epoch, ltot / batch_idx)

            if SAVE_FILE_PRETRAIN and (ltot / batch_idx) < best_prtr_val_loss:
                saver.save(sess, SAVE_FILE_PRETRAIN + 'b')
                best_prtr_val_loss = (ltot / batch_idx)
            if SAVE_FILE_PRETRAIN:
                saver.save(sess, SAVE_FILE_PRETRAIN)

    counter = 1
    for epoch in xrange(N_EPOCHS):
        batch_idx = 0
        for batch in c.get_train_batch(BATCH_SIZE):
            batch_idx += 1

            # To remove start token, since generator doesn't generate it either
            batch = c.convert_batch_to_input_target(batch)
            _, batch_targets = batch

            if batch_idx % NUM_G == 0:
                z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                c_z, h_z = np.vsplit(z, 2)
                _, d_loss_curr, summary_str = sess.run([d_optim, d_loss, d_loss_sum], feed_dict={
                    inputs: batch_targets,
                    initial_c: c_z,
                    initial_h: h_z
                })
                writer.add_summary(summary_str, counter)

            if batch_idx % NUM_D != 0:
                continue

            z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
            c_z, h_z = np.vsplit(z, 2)
            _, g_loss_curr, summary_str = sess.run([g_optim, g_loss, g_loss_sum], feed_dict={
                initial_c: c_z,
                initial_h: h_z
            })
            writer.add_summary(summary_str, counter)

            counter += 1
            #print "Epoch: [%d] Batch: %d d_loss: %.8f, g_loss: %.8f" % (epoch, batch_idx, d_loss_curr, g_loss_curr)

        if DATASET == 'oracle':
            generate_test_file(g_test, sess, t.eval_file)
            print "NLL Oracle Loss after epoch %d: %.8f" % (epoch, t.get_loss())
        else:
            batch_idx = 0
            ltot = 0.
            for batch in c.get_valid_batch(BATCH_SIZE):
                batch = c.convert_batch_to_input_target(batch)
                batch_input, batch_targets = batch

                z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                c_z, h_z = np.vsplit(z, 2)
                g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                    inputs_pre: batch_input,
                    initial_c: c_z,
                    initial_h: h_z,
                    targets: batch_targets
                })
                ltot += g_pre_loss_curr
                writer.add_summary(summary_str, counter)
                batch_idx += 1
            print "Valid Loss after epoch %d: %.8f" % (epoch, ltot / batch_idx)
            batch_idx = 0
            ltot = 0.
            for batch in c.get_test_batch(BATCH_SIZE):
                batch = c.convert_batch_to_input_target(batch)
                batch_input, batch_targets = batch

                z = sample_Z(BATCH_SIZE * 2, HIDDEN_STATE_SIZE)
                c_z, h_z = np.vsplit(z, 2)
                g_pre_loss_curr, summary_str = sess.run([g_pre_loss, g_pre_loss_sum], feed_dict={
                    inputs_pre: batch_input,
                    initial_c: c_z,
                    initial_h: h_z,
                    targets: batch_targets
                })
                ltot += g_pre_loss_curr
                writer.add_summary(summary_str, counter)
                batch_idx += 1
            print "Test Loss after epoch %d: %.8f" % (epoch, ltot / batch_idx)

        samples = []
        for i in xrange(10):
            samples.extend(generate_samples())
            with open('samples_{}.txt'.format(epoch), 'w') as f:
                for s in samples:
                    s = "".join(s)
                    f.write(s + "\n")
