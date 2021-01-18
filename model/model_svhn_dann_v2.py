from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import tensorflow as tf
import ops
slim = tf.contrib.slim
from datetime import datetime
import os.path
import time
import numpy as np
from flip_gradient import flip_gradient
import math


class DANN(object):
    def __init__(self, config):
        self.gpu_fraction = config.fraction
        self.src_name = config.source_name
        self.trg_name = config.target_name
        self.is_dan = config.is_dan
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.source_data_dir = config.source_data_dir
        self.target_data_dir = config.target_data_dir
        self.source_name = config.source_name
        self.target_name = config.target_name
        self.saver_path = config.saver_path
        self.max_steps = config.max_steps
        self.decay_steps = config.decay_steps
        self.decay_factor = config.decay_factor
        self.beta = config.beta
        self.alpha = config.alpha
        self.d_iter = config.d_iter
        self.gpu_num = config.gpu_num

    def feature_extractor(self, net):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=1, padding="SAME"):
                with slim.arg_scope([slim.max_pool2d], kernel_size=[3, 3], stride=2, padding="SAME"):
                    with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                        activation_fn=tf.nn.relu, is_training=True):
                        # 32, 32, 3 -> 16, 16, 64
                        net = slim.conv2d(net, 64, scope='conv1')
                        net = slim.max_pool2d(net, stride=2, scope='pool1')
                        net = slim.batch_norm(net, scope='bn1')
                        # 16, 16, 64 -> 8, 8, 64
                        net = slim.conv2d(net, 64, scope='conv2')
                        net = slim.max_pool2d(net, stride=2, scope='pool2')
                        net = slim.batch_norm(net, scope='bn2')
                        # 8, 8, 64 -> 8, 8, 128 (8192)
                        net = slim.conv2d(net, 128, scope='conv3')
                        # net = slim.max_pool2d(net, stride=2, scope='pool3')
                        # net = slim.batch_norm(net, scope='bn3')
                        net = tf.contrib.layers.flatten(net)
                        return net

    def domain_classifier(self, inputs, global_step):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                activation_fn=tf.nn.relu, is_training=True):
                p = tf.cast(global_step, tf.float32) / 30000
                decay = 0.5 * (2 / (1. + tf.exp(-2.5 * p)) - 1) + 0.1
                grl = flip_gradient(inputs, decay)

                # [2048]---[1024]
                net = slim.fully_connected(grl, 1024, scope="fc0")
                net = slim.batch_norm(net, scope='bn0')

                net = slim.fully_connected(net, 1024, scope="fc1")
                net = slim.batch_norm(net, scope='bn1')
                # [1024]---[1]
                logits = slim.fully_connected(net, 1, scope="fc2")

                source_logits = tf.slice(logits, [0, 0], [self.batch_size, 1])
                target_logits = tf.slice(logits, [self.batch_size, 0], [self.batch_size, 1])

                # compute domain_loss
                source_domain_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=source_logits, labels=tf.ones_like(source_logits)
                    ))
                target_domain_loss = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(
                        logits=target_logits, labels=tf.zeros_like(target_logits)
                    ))
                domain_loss = (source_domain_loss + target_domain_loss) / 2.0

                # compute domain_acc
                source_domain_correct = source_logits > 0
                target_domain_correct = target_logits < 0
                domain_correct = tf.concat([source_domain_correct, target_domain_correct], 0)
                domain_acc = tf.reduce_mean(tf.cast(domain_correct, tf.float32))

        return domain_loss, domain_acc, decay

    def label_classifier(self, inputs):
        # classifier network
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                activation_fn=tf.nn.relu, is_training=True):
                # source_classifier
                net = slim.fully_connected(inputs, 3072, scope="fc0")
                net = slim.batch_norm(net, scope='bn0')
                # fc1
                net = slim.fully_connected(net, 2048, scope="fc1")
                net = slim.batch_norm(net, scope='bn1')
                logits = slim.fully_connected(net, 10, scope="fc2")
                return logits

    def build_model(self, source_images, target_images, source_labels, target_labels, step, is_train=True):
        if not is_train:
            single_feature = self.feature_extractor(target_images)
            target_logits = self.label_classifier(single_feature)
            target_correct = tf.equal(tf.argmax(target_logits, 1), tf.argmax(target_labels, 1))
            target_acc = tf.reduce_mean(tf.cast(target_correct, tf.float32))
            return target_acc

        merged_images = tf.concat([source_images, target_images], 0)
        with tf.variable_scope("feature_extractor") as scope:
            single_feature = self.feature_extractor(merged_images)

        with tf.variable_scope("label_classifier") as scope:
            logits = self.label_classifier(single_feature)
            source_logits = tf.slice(logits, [0, 0], [self.batch_size, 10])
            target_logits = tf.slice(logits, [self.batch_size, 0], [self.batch_size, 10])

        with tf.variable_scope("domain_classifier") as scope:
            domain_loss, domain_acc, decay = self.domain_classifier(single_feature, step)

        # compute label_loss
        # source data labeled
        source_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=source_logits, labels=source_labels))

        label_loss = source_label_loss  # + self.beta * target_label_loss

        # source_acc and target acc
        source_correct = tf.equal(tf.argmax(source_logits, 1), tf.argmax(source_labels, 1))
        source_acc = tf.reduce_mean(tf.cast(source_correct, tf.float32))

        target_correct = tf.equal(tf.argmax(target_logits, 1), tf.argmax(target_labels, 1))
        target_acc = tf.reduce_mean(tf.cast(target_correct, tf.float32))

        total_loss = domain_loss + label_loss

        slim_loss = slim.losses.get_total_loss()
        total_loss = total_loss + slim_loss

        lr = tf.train.exponential_decay(self.learning_rate, step, self.decay_steps,
                                        self.decay_factor,
                                        staircase=True)

        train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(total_loss, step)

        pack_1 = [train_op, total_loss, domain_loss, label_loss, source_label_loss]
        pack_2 = [source_acc, target_acc, slim_loss, decay]

        return pack_1, pack_2

    def train_model(self):
        with tf.Graph().as_default():
            self.global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            source_images, source_labels = ops.load_batch_svhn(
                self.batch_size,
                self.source_data_dir,
                'train.' + self.source_name,
                is_train=True
            )
            target_images, target_labels = ops.load_batch_mnist(
                self.batch_size,
                self.target_data_dir,
                'train.' + self.target_name,
                is_train=True, is_rgb=True,
                data_type=tf.uint8,
                size=32,
                shuffle=True
            )
            pack_1, pack_2 = self.build_model(source_images, target_images,
                                              source_labels, target_labels,
                                              self.global_step)

            model_train_op = pack_1[0]
            model_total_loss = pack_1[1]
            model_domain_loss = pack_1[2]
            model_label_loss = pack_1[3]
            model_s_label_loss = pack_1[4]

            model_source_acc = pack_2[0]
            model_target_acc = pack_2[1]
            model_slim_loss = pack_2[2]
            decay = pack_2[3]

            # model_source_image = self.source_images
            # model_target_image = self.target_images

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_num
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = .5
            sess = tf.Session(config=config)
            init = tf.global_variables_initializer()
            sess.run(init)

            tf.train.start_queue_runners(sess=sess)
            for step in xrange(self.max_steps):
                start_time = time.time()
                _ = sess.run(model_train_op)
                t_loss, d_loss, l_loss = sess.run((model_total_loss, model_domain_loss, model_label_loss))
                duration = time.time() - start_time

                assert not np.isnan(t_loss), 'Model diverged with loss = NaN'
                if step % 200 == 0:
                    num_examples_per_step = self.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = (
                        'step %d, total_loss = %.3f domain_loss=%.3f label_loss =%.3f (%.1f examples/sec; %.3f '
                        'sec/batch)')
                    print(format_str % (step, t_loss, d_loss, l_loss,
                                        examples_per_sec, sec_per_batch))

                    l_loss_s, acc_s, acc_t = sess.run(
                        (model_s_label_loss, model_source_acc, model_target_acc))

                    format_str_1 = 'src_loss:' + str(l_loss_s) + ' ' + 'trg_loss:'\
                                   + 'src_acc:' + str(acc_s) + ' ' + 'trg_acc:' + str(acc_t)
                    print(format_str_1)
                    print('decay:', sess.run(decay))
                if step % 200 == 0 or (step + 1) == self.max_steps:
                    save_path = os.path.join(self.saver_path, 'dann_model')
                    saver.save(sess, save_path, global_step=step)

    def test_model(self, step):
        with tf.Graph().as_default():
            self.global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            test_target_images, test_target_labels = ops.load_batch_mnist(
                batch_size=1000,
                data_dir=self.target_data_dir,
                dataname='test.' + self.target_name,
                is_train=False,
                size=32,
                is_rgb=True
            )
            # self.test_source_images, self.test_source_labels = self.load_batch_usps_test(self.source_data_dir)
            # self.test_target_images, self.test_target_labels = self.load_batch_mnist_test(self.target_data_dir)

            trg_acc = self.build_model(None, test_target_images,
                                       None, test_target_labels,
                                       self.global_step, False)
            model_target_acc = trg_acc
            saver = tf.train.Saver(tf.global_variables())
            os.environ['CUDA_VISIBLE_DEVICES'] = self.gpu_num
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = .5
            sess = tf.Session(config=config)
            restore_path = os.path.join(self.saver_path, 'dann_model-' + str(step))
            saver.restore(sess, restore_path)
            precision = self.eval_once(sess, model_target_acc)

        return precision

    def eval_once(self, sess, accuracy):
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(10000 / self.batch_size))
            true_count = 0
            step = 0
            while step < num_iter and not coord.should_stop():
                predictions = sess.run([accuracy])
                true_count += np.sum(predictions)
                step += 1
            precision = (1.0 * true_count) / num_iter
            print('precision:', precision)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=1)

        return precision
