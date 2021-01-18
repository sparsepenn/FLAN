from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import re
import sys
import tensorflow as tf
import ops

slim = tf.contrib.slim
import tensorflow.contrib.layers as ly
from datetime import datetime
import os.path
import time
import numpy as np
from flip_gradient import flip_gradient
import math

sys.path.append(r"/home/fang/downloads/models/research/slim");
from nets import resnet_v1
from nets import resnet_utils


class DJARTN(object):
    def __init__(self, config):
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.source_data_dir = config.source_data_dir
        self.target_data_dir = config.target_data_dir
        self.saver_path = config.saver_path
        self.image_save_dir = config.image_save_dir
        self.max_steps = config.max_steps
        self.decay_steps = config.decay_steps
        self.decay_factor = config.decay_factor
        self.beta = config.beta

        ##add
        self.image_size = config.image_size
        self.crop_image_size = config.crop_image_size
        self.image_depth = config.image_depth
        self.minimal_queue = config.minimal_queue
        self.Resnet_pretrain_checkpoint = config.Resnet_pretrain_checkpoint
        self.source_name = config.source_name
        self.target_name = config.target_name
        self.checkpoint_name = config.checkpoint_name
        self.amazon_size = config.amazon_size
        self.dslr_size = config.dslr_size
        self.webcam_size = config.webcam_size

    def load_batch(self, data_dir, data_name, is_train=True):
        filename = os.path.join(data_dir, data_name)
        if is_train:
            images, labels = ops.load_images_labels_train(filename, self.batch_size, self.image_size, self.image_depth,
                                                          self.crop_image_size, self.minimal_queue)
        else:
            images, labels = ops.load_images_labels_test(filename, self.batch_size, self.image_size, self.image_depth,
                                                         self.crop_image_size, self.minimal_queue)

        images = tf.cast(images, tf.float32)

        labels = tf.cast(labels, tf.int32)
        labels = tf.reshape(labels, [self.batch_size])
        labels = tf.one_hot(labels, 31)

        return images, labels

    def feature_extractor(self, inputs):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.001)):
            with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                activation_fn=tf.nn.relu, is_training=True):
                # [100]---[50]
                # net_layer_1 = slim.fully_connected(inputs,50,scope="fc1")
                net_layer_1 = inputs
                net_layer_1 = slim.batch_norm(net_layer_1, scope='bn1')
                # [50]---[31]no activation function
                net_layer_2 = slim.fully_connected(net_layer_1, 31, scope="fc2")

        return net_layer_2

    def domain_classifier(self, inputs, global_step):
        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                activation_fn=tf.nn.relu, is_training=True):
                p = tf.cast(global_step, tf.float32) / 30000
                decay = 1.8 * (2 / (1. + tf.exp(-5.0 * p)) - 1)
                grl = flip_gradient(inputs, decay)

                # [961]---[256]
                net = slim.fully_connected(grl, 256, scope="fc0")
                net = slim.batch_norm(net, scope='bn0')

                # [256,64]
                net = slim.fully_connected(net, 64, scope="fc1")
                net = slim.batch_norm(net, scope='bn1')

                # [256,64]
                net = slim.fully_connected(net, 64, scope="fc2")
                net = slim.batch_norm(net, scope='bn2')

                # [64]---[1]
                logits = slim.fully_connected(net, 1, scope="fc3")

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
                source_domain_acc = tf.reduce_mean(tf.cast(source_domain_correct, tf.float32))

        return domain_loss, domain_acc, decay, source_domain_acc

    def label_classifier(self, inputs):
        # classifier network
        source_inputs = tf.slice(inputs, [0, 0], [self.batch_size, 31])
        target_inputs = tf.slice(inputs, [self.batch_size, 0], [self.batch_size, 31])

        with slim.arg_scope([slim.fully_connected],
                            weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                            weights_regularizer=slim.l2_regularizer(0.0001)):
            with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                activation_fn=tf.nn.relu, is_training=True):
                # source_classifier
                # fc0
                net = slim.fully_connected(source_inputs, 31, scope="fc0")
                net = slim.batch_norm(net, scope='bn0')

                # fc1
                source_logits = slim.fully_connected(net, 31, scope="fc1")
                delta_loss = tf.nn.l2_loss(source_logits)
                loss = tf.nn.l2_loss(source_inputs)

                ratio = delta_loss / loss
                source_logits = tf.add(source_inputs, source_logits)
                target_logits = target_inputs

        return source_logits, target_logits, ratio

    def build_model(self, source_images, target_images, source_labels, target_labels, step):

        merged_images = tf.concat([source_images, target_images], 0)

        ####load pretrain model
        with slim.arg_scope(resnet_utils.resnet_arg_scope()):
            logits, end_points = resnet_v1.resnet_v1_50(merged_images, num_classes=64, is_training=True)

        # for var in slim.get_model_variables():
        #    print(var.name)

        scope_to_train = ['resnet_v1_50/logits/weights:0', 'resnet_v1_50/logits/biases:0']
        variables_to_restore = []

        for var in slim.get_model_variables():
            if var.name in scope_to_train:
                print(var.name)
            else:
                variables_to_restore.append(var)
                # print('to_restore:',var.name)

        variables_to_train = []
        scopes = [scope.strip() for scope in scope_to_train]
        for scope in scopes:
            variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
            variables_to_train.extend(variables)

        checkpoint_path = os.path.join(self.Resnet_pretrain_checkpoint, self.checkpoint_name)
        init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

        ####fine tune model
        with tf.variable_scope("feature_extractor") as scope:
            single_feature = self.feature_extractor(logits)
        with tf.variable_scope("label_classifier") as scope:
            source_logits, target_logits, ratio = self.label_classifier(single_feature)

        # Kronecker product of multi_feature and label
        # for Unsupervisied domain adaption
        merged_labels = tf.concat([source_labels, tf.nn.softmax(target_logits)], 0)

        stack_1 = tf.stack([single_feature for i in range(31)], axis=2)
        stack_1 = tf.reshape(stack_1, [2 * self.batch_size, 961])
        stack_2 = tf.stack([merged_labels for i in range(31)], axis=1)
        stack_2 = tf.reshape(stack_2, [2 * self.batch_size, 961])

        Joint_layer_feature = tf.multiply(stack_1, stack_2)
        with tf.variable_scope("domain_classifier") as scope:
            domain_loss, domain_acc, decay, source_domain_acc = self.domain_classifier(Joint_layer_feature, step)

        # compute label_loss
        # source data labeled
        source_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=source_logits, labels=source_labels))

        # target data not labeled(cross entropy with its self)
        # labels =tf.nn.softmax(target_labels)
        target_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=target_logits, labels=tf.nn.softmax(target_logits)))

        label_loss = source_label_loss + self.beta * target_label_loss

        # Resnet last layer net---> weight regular
        regularization_loss = tf.losses.get_regularization_loss(scope='InceptionV1/Logits')

        t_vars = tf.trainable_variables()
        var_feature_extractor = [var for var in t_vars if 'feature_extractor' in var.name]
        var_domain_classifier = [var for var in t_vars if 'domain_classifier' in var.name]
        var_label_classifier = [var for var in t_vars if 'label_classifier' in var.name]

        # transfer learning model network---> weight regular
        ##add train variables
        for var in var_feature_extractor:
            regularization_loss = regularization_loss + tf.losses.get_regularization_loss(scope=var.name)
            variables_to_train.append(var)

        for var in var_domain_classifier:
            regularization_loss = regularization_loss + tf.losses.get_regularization_loss(scope=var.name)
            variables_to_train.append(var)

        for var in var_label_classifier:
            regularization_loss = regularization_loss + tf.losses.get_regularization_loss(scope=var.name)
            variables_to_train.append(var)

        ##obtain variables_to_finetune
        variables_to_fine_tune = []
        t_vars = tf.trainable_variables()
        for var in t_vars:
            if "resnet_v1_50/block3" in var.name:
                variables_to_fine_tune.append(var)
            elif "resnet_v1_50/block4" in var.name:
                variables_to_fine_tune.append(var)

        # for var in variables_to_fine_tune :
        #    print("variables_to_fine_tune",var)

        for var in variables_to_fine_tune:
            regularization_loss = regularization_loss + tf.losses.get_regularization_loss(scope=var.name)

        total_loss = label_loss + 10 * regularization_loss + domain_loss
        #        total_loss = label_loss + domain_loss
        ##compute acc
        source_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(source_logits, 1), tf.argmax(source_labels, 1)), tf.float32))
        target_acc = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(target_logits, 1), tf.argmax(target_labels, 1)), tf.float32))

        ##obtain op
        lr = tf.train.exponential_decay(self.learning_rate, step, self.decay_steps,
                                        self.decay_factor,
                                        staircase=True)
        # test
        # for var in variables_to_train:
        #    print('variables_to_train:',var)

        train_op = tf.train.GradientDescentOptimizer(lr).minimize(total_loss, step, var_list=variables_to_train)
        fine_tune_op = tf.train.GradientDescentOptimizer(0.1 * lr).minimize(total_loss, global_step=None,
                                                                            var_list=variables_to_fine_tune)

        pack_1 = [train_op, total_loss, domain_loss, label_loss, source_label_loss, target_label_loss, fine_tune_op]
        pack_2 = [source_acc, target_acc, domain_acc, decay, ratio]

        return pack_1, pack_2, init_fn, source_domain_acc

    def train_model(self):
        with tf.Graph().as_default():
            self.global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            self.source_images, self.source_labels = self.load_batch(self.source_data_dir, self.source_name,
                                                                     is_train=True)
            self.target_images, self.target_labels = self.load_batch(self.target_data_dir, self.target_name,
                                                                     is_train=True)

            pack_1, pack_2, init_fn, source_domain_acc = self.build_model(self.source_images, self.target_images,
                                                                          self.source_labels, self.target_labels,
                                                                          self.global_step)

            model_train_op = pack_1[0]
            model_total_loss = pack_1[1]
            model_domain_loss = pack_1[2]
            model_label_loss = pack_1[3]
            model_s_label_loss = pack_1[4]
            model_t_label_loss = pack_1[5]
            model_fine_tune_op = pack_1[6]

            model_source_acc = pack_2[0]
            model_target_acc = pack_2[1]
            model_domain_acc = pack_2[2]
            decay = pack_2[3]
            ratio = pack_2[4]

            model_source_image = self.source_images
            model_target_image = self.target_images

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.4
            with tf.Session(config=config) as sess:
                init = tf.global_variables_initializer()
                sess.run(init)
                init_fn(sess)
                tf.train.start_queue_runners(sess=sess)
                for step in range(self.max_steps):
                    start_time = time.time()
                    _ = sess.run(model_train_op)
                    _ = sess.run(model_fine_tune_op)
                    # t_loss,l_loss =sess.run((model_total_loss,model_label_loss))
                    duration = time.time() - start_time
                    # assert not np.isnan(t_loss), 'Model diverged with loss = NaN'
                    if step % 10 == 0:
                        num_examples_per_step = self.batch_size
                        examples_per_sec = num_examples_per_step / duration
                        sec_per_batch = float(duration)
                        t_loss, l_loss, d_loss = sess.run((model_total_loss, model_label_loss, model_domain_loss))
                        format_str = (
                        '%s: step %d, total_loss = %.3f label_loss =%.3f domain_loss=%.3f(%.1f examples/sec; %.3f '
                        'sec/batch)')
                        print(format_str % (datetime.now(), step, t_loss, l_loss, d_loss,
                                            examples_per_sec, sec_per_batch))

                        source_acc, target_acc, domain_acc = sess.run(
                            (model_source_acc, model_target_acc, model_domain_acc))
                        format_str_1 = 'src_acc:' + str(source_acc) + ' ' + 'trg_acc:' + str(
                            target_acc) + ' ' + 'domain_acc:' + str(domain_acc)
                        print(format_str_1)

                        source_label_loss, target_label_loss = sess.run((model_s_label_loss, model_t_label_loss))
                        format_str_2 = 'source_label_loss:' + str(source_label_loss) + ' ' + 'target_label_loss:' + str(
                            target_label_loss)
                        print(format_str_2)

                        decay_, ratio_ = sess.run((decay, ratio))
                        format_str_3 = 'decay:' + str(decay_) + ' ' + 'ratio:' + str(ratio_)
                        print(format_str_3)
                        s_acc = sess.run(source_domain_acc)
                        print("source_domian_acc:" + str(s_acc))
                        # if step %10 ==0:
                        #    s_images,t_images =sess.run((model_source_image,model_target_image))
                        # save images to image_save_dir
                    #    src_dir = os.path.join(self.image_save_dir , 'source_images__'+str(step/10)+'.jpg')
                    #    trg_dir = os.path.join(self.image_save_dir , 'target_images__'+str(step/10)+'.jpg')
                    #    ops.show_result(s_images,src_dir,self.crop_image_size,self.image_depth)
                    #    ops.show_result(t_images,trg_dir,self.crop_image_size,self.image_depth)
                    if step % 100 == 0 or (step + 1) == self.max_steps:
                        save_path = os.path.join(self.saver_path, 'Resnet_model')
                        saver.save(sess, save_path, global_step=step)

    def test_model(self, step):
        with tf.Graph().as_default():
            self.batch_size = 100
            self.global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            self.source_images, self.source_labels = self.load_batch(self.source_data_dir, self.source_name,
                                                                     is_train=True)
            self.target_images, self.target_labels = self.load_batch(self.target_data_dir, self.target_name,
                                                                     is_train=False)

            pack_1, pack_2, init_fn, source_acc = self.build_model(self.source_images, self.target_images,
                                                                   self.source_labels, self.target_labels,
                                                                   self.global_step)

            model_source_acc = pack_2[0]
            model_target_acc = pack_2[1]
            saver = tf.train.Saver(tf.global_variables())
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = 0.4

            sess = tf.Session(config=config)
            restore_path = os.path.join(self.saver_path, 'Resnet_model-' + str(step))
            saver.restore(sess, restore_path)
            precision = self.eval_once(sess, model_target_acc)
            s_acc = sess.run(source_acc)
            print("source_domain__acc:" + str(s_acc))

        return precision

    def eval_once(self, sess, accuracy):
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(self.amazon_size / self.batch_size) + 1
            true_count = 0
            step = 0
            while step < num_iter:
                predictions = sess.run([accuracy])
                true_count += np.sum(predictions)
                step += 1
                print("step:" + str(step) + " prediction:" + str(predictions))
            precision = (1.0 * true_count) / num_iter
            print('precision:', precision)


        except Exception as e:
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=1)

        return precision





























