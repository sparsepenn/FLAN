from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import ops
import params
import tensorflow.contrib.slim as slim
from datetime import datetime
import os.path
import time


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
        self.saver_path = config.saver_path
        self.max_steps = config.max_steps
        self.decay_steps = config.decay_steps
        self.decay_factor = config.decay_factor
        self.beta = config.beta
        self.alpha = config.alpha

    def kron(self, f_1, dim_1, f_2, dim_2):
        stack_1 = tf.stack([f_1 for _ in range(dim_1)], axis=2)
        stack_1 = tf.reshape(stack_1, [-1, dim_1 * dim_2])
        stack_2 = tf.stack([f_2 for _ in range(dim_2)], axis=1)
        stack_2 = tf.reshape(stack_2, [-1, dim_1 * dim_2])
        joint_layer_feature = tf.multiply(stack_1, stack_2)
        return joint_layer_feature

    def generator(self, inputs, reuse=False):
        with tf.variable_scope("generator", reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                with slim.arg_scope([slim.conv2d], kernel_size=[5, 5], stride=1, padding="SAME"):
                    with slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=2, padding="SAME"):
                        with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                            activation_fn=tf.nn.relu, is_training=True):
                            # [28,28,1]---[14,14,64]
                            net = slim.conv2d(inputs, 32, scope='conv0')
                            net = slim.max_pool2d(net, scope='pool0')
                            net = slim.batch_norm(net, scope='bn0')
                            # [14,14,64]---[7,7,128]
                            net = slim.conv2d(net, 64, scope='conv1')
                            net = slim.max_pool2d(net, scope='pool1')
                            net = slim.batch_norm(net, scope='bn1')
                            # [7,7,128]---[7,7,128]
                            net = slim.conv2d(net, 48, scope='conv2')
                            net = slim.batch_norm(net, scope='bn2')

                            # reshape[7,7,128]---[7*7*128]
                            net = tf.reshape(net, [-1, 7 * 7 * 48])
                            # [7*7*128]---[100]
                            net = slim.fully_connected(net, 100, scope="fc3")
                            net = slim.batch_norm(net, scope='bn3')
                            # [100]---[10]no activation function
                            net = slim.fully_connected(net, 10, scope="fc4")

                            return net

    def discriminator(self, inputs, reuse=False):
        with tf.variable_scope("discriminator", reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                    activation_fn=tf.nn.relu, is_training=True):
                    # [100]---[100]
                    net = slim.fully_connected(inputs, 100, scope="fc0")
                    net = slim.batch_norm(net, scope='bn0')

                    # [128]---[1]
                    logits = slim.fully_connected(net, 1, scope="fc1")
                    return logits

    def classifier(self, inputs, reuse=False):
        # classifier network
        with tf.variable_scope("classifier", reuse=reuse) as scope:
            with slim.arg_scope([slim.fully_connected],
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.02), activation_fn=None,
                                weights_regularizer=slim.l2_regularizer(0.0001)):
                with slim.arg_scope([slim.batch_norm], decay=0.80, center=True, scale=True, epsilon=1e-5,
                                    activation_fn=tf.nn.relu, is_training=True):
                    # fc0
                    net = slim.fully_connected(inputs, 10, scope="fc0")
                    net = slim.batch_norm(net, scope='bn0')
                    # fc1
                    logits = slim.fully_connected(net, 10, scope="fc1")
                    logits = tf.add(inputs, logits)

                    return logits

    def build_model(self, source_images, target_images, source_labels, target_labels, step, is_train=True):
        if not is_train:
            single_feature = self.generator(target_images)
            target_logits = single_feature
            target_correct = tf.equal(tf.argmax(target_logits, 1), tf.argmax(target_labels, 1))
            target_acc = tf.reduce_mean(tf.cast(target_correct, tf.float32))
            return target_acc
        # if seperate generator with reuse, acc down!!! So, merge it... maybe the parameter step?
        merged_images = tf.concat([source_images, target_images], 0)
        merege_features = self.generator(merged_images)
        # method one: not share classifier
        source_feature = tf.slice(merege_features, [0, 0], [self.batch_size, 10])
        target_feature = tf.slice(merege_features, [self.batch_size, 0], [self.batch_size, 10])
        source_logits = self.classifier(source_feature)
        target_logits = target_feature
        # methed two: share classifier
        # c_logits = self.classifier(merege_features)
        # source_logits = tf.slice(c_logits, [0, 0], [self.batch_size, 10])
        # target_logits = tf.slice(c_logits, [self.batch_size, 0], [self.batch_size, 10])

        # ratio: in order to check if f_s approximate f_t. Here ratio should be smaller(0.02)
        delta_loss = tf.nn.l2_loss(source_logits)
        loss = tf.nn.l2_loss(source_feature)
        ratio = delta_loss / loss
        ####################################################
        merged_labels = tf.concat([tf.nn.softmax(source_logits), tf.nn.softmax(target_logits)], 0)
        Joint_layer_feature = self.kron(merege_features, 10, merged_labels, 10)

        d_logits = self.discriminator(Joint_layer_feature)
        d_source_logits = tf.slice(d_logits, [0, 0], [self.batch_size, 1])
        d_target_logits = tf.slice(d_logits, [self.batch_size, 0], [self.batch_size, 1])

        # compute domain_loss
        source_domain_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_source_logits, labels=tf.ones_like(d_source_logits)
            ))
        target_domain_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=d_target_logits, labels=tf.zeros_like(d_target_logits)
            ))
        domain_loss = (source_domain_loss + target_domain_loss) / 2.0

        # compute label_loss
        # source data labeled
        source_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=source_logits, labels=source_labels))

        # target data not labeled(cross entropy with its self)
        target_label_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=target_logits, labels=tf.nn.softmax(target_logits)))

        c_loss = source_label_loss + self.beta * target_label_loss

        # source_acc and target acc
        source_correct = tf.equal(tf.argmax(source_logits, 1), tf.argmax(source_labels, 1))
        source_acc = tf.reduce_mean(tf.cast(source_correct, tf.float32))

        target_correct = tf.equal(tf.argmax(target_logits, 1), tf.argmax(target_labels, 1))
        target_acc = tf.reduce_mean(tf.cast(target_correct, tf.float32))

        lr = tf.train.exponential_decay(self.learning_rate, step, self.decay_steps,
                                        self.decay_factor,
                                        staircase=True)
        p = tf.cast(step, tf.float32) / self.max_steps
        decay = 0.17 * (2 / (1. + tf.exp(-2.5 * p)) - 1) + 0.1

        # regularization_loss = slim.losses.get_total_loss()
        regularization_loss = tf.add_n(tf.losses.get_regularization_losses())
        d_loss = domain_loss + regularization_loss
        g_loss = -decay*domain_loss + c_loss
        ##################
        # train_op
        ##################
        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'discriminator' in var.name]
        g_vars = [var for var in t_vars if 'generator' in var.name]
        c_vars = [var for var in t_vars if 'classifier' in var.name]

        d_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(d_loss, step, d_vars)
        g_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(g_loss, step, g_vars)
        c_train_op = tf.train.MomentumOptimizer(lr, 0.9).minimize(c_loss, step, c_vars)

        pack_op_loss = [d_train_op, g_train_op, c_train_op, d_loss, c_loss]
        pack_acc = [source_acc, target_acc, decay, lr, ratio]

        return pack_op_loss, pack_acc

    def train_model(self):
        with tf.Graph().as_default():
            global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            source_images, source_labels = ops.load_batch_mnist(self.batch_size,
                                                                self.source_data_dir,
                                                                dataname='train.' + self.src_name,
                                                                is_train=True)
            target_images, target_labels = ops.load_batch_usps(self.batch_size,
                                                               self.target_data_dir,
                                                               dataname='train.' + self.trg_name,
                                                               is_train=True)
            # source_images, source_labels = ops.load_batch_mnist(self.batch_size, self.source_data_dir,
            #                                                     dataname='train.mnist',
            #                                                     is_RGB=True)
            # target_images, target_labels = ops.load_batch_mnistm(self.batch_size, self.target_data_dir,
            #                                                      dataname='train.mnistm')
            pack_op_loss, pack_acc = self.build_model(source_images, target_images,
                                                      source_labels, target_labels,
                                                      global_step, is_train=True)

            d_train_op = pack_op_loss[0]
            g_train_op = pack_op_loss[1]
            c_train_op = pack_op_loss[2]

            d_loss = pack_op_loss[3]
            c_loss = pack_op_loss[4]

            source_acc = pack_acc[0]
            target_acc = pack_acc[1]
            decay = pack_acc[2]
            lr = pack_acc[3]
            ratio = pack_acc[4]

            saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)

            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = .3
            sess = tf.Session(config=config)
            # set_session(sess)
            init = tf.global_variables_initializer()
            sess.run(init)
            sess.run(tf.local_variables_initializer())

            tf.train.start_queue_runners(sess=sess)
            for step in xrange(self.max_steps + 1):
                start_time = time.time()
                _, _, _ = \
                    sess.run([d_train_op, g_train_op, c_train_op])
                duration = time.time() - start_time
                # assert not np.isnan(t_loss), 'Model diverged with loss = NaN'
                if step % params.train_show == 0:
                    d_l, c_l = sess.run([d_loss, c_loss])
                    acc_s, acc_t = sess.run([source_acc, target_acc])
                    examples_per_sec = self.batch_size / duration

                    format_str = (
                        '%s: step %d, d_loss = %.3f c_loss=%.3f [%.1f examples/sec]'
                    )
                    print(format_str % (datetime.now(), step, d_l, c_l,
                                        examples_per_sec))

                    format_str_1 = 'acc_s: {:.3f}, acc_t: {:.3f}' \
                        .format(acc_s, acc_t)
                    print(format_str_1)
                    print('decay:', sess.run(decay))
                    print('lr: ', sess.run(lr))
                    print('ratio: ', sess.run(ratio))

                if step % params.train_save == 0:
                    save_path = os.path.join(self.saver_path, params.save_model)
                    saver.save(sess, save_path, global_step=step)

    def test_model(self, step):
        with tf.Graph().as_default():
            global_step = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)
            test_target_images, test_target_labels = ops.load_batch_usps(batch_size=2007,
                                                                         data_dir=self.target_data_dir,
                                                                         dataname='test.' + self.trg_name,
                                                                         is_train=False)
            # test_target_images, test_target_labels = ops.load_batch_mnistm(self.batch_size,
            #                                                                self.target_data_dir,
            #                                                                'test.mnistm')

            target_acc = self.build_model(None, test_target_images,
                                          None, test_target_labels,
                                          global_step, is_train=False)

            saver = tf.train.Saver(tf.global_variables())
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            config = tf.ConfigProto()
            config.gpu_options.per_process_gpu_memory_fraction = .3
            sess = tf.Session(config=config)
            restore_path = os.path.join(self.saver_path, params.save_model + '-' + str(step))
            print(restore_path)
            sess.run(tf.local_variables_initializer())
            saver.restore(sess, restore_path)
            precision = self.eval_once(sess, target_acc)

        return precision

    def eval_once(self, sess, accuracy):
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            true_count = 0.0
            num = 0
            while not coord.should_stop():
                predictions = sess.run(accuracy)
                # print ('predictions: ', predictions)
                true_count += predictions
                num += 1
        except Exception as e:
            print('num: ', num)
            print('true_count: ', true_count)
            precision = true_count / num
            print('precision:', precision)
            coord.request_stop(e)
            coord.request_stop()
            coord.join(threads, stop_grace_period_secs=1)
            return precision
