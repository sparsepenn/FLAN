import tensorflow as tf
import numpy as np
from model import DJARTN
import scipy.io as sio
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags
flags.DEFINE_integer("max_steps", 30000, "maximum step to train model")
flags.DEFINE_integer("decay_steps", 3000, "steps to change learning rate")
flags.DEFINE_float("learning_rate", 0.02, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("decay_factor", 0.92, "the change of ration of learning rate")
flags.DEFINE_float("beta", 0.6, "the ratio of target label loss and source label loss")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [64]")
flags.DEFINE_string("source_data_dir", "./data/", "the path of source data")
flags.DEFINE_string("target_data_dir", "./data/", "the path of target data")
flags.DEFINE_string("saver_path", "./saver", "Directory name to save the checkpoints [checkpoint]")
######### add
flags.DEFINE_integer("image_size", 256, "the size of source images and target images")
flags.DEFINE_integer("crop_image_size", 224, "the size of crop of source images and target images")
flags.DEFINE_integer("image_depth", 3, "the channels of source images and target images")
flags.DEFINE_integer("minimal_queue", 1000, "the minimal size of cache queue")
flags.DEFINE_string("Resnet_pretrain_checkpoint", "./pretrain_model",
                    "the path of checkpoint of resnet pretrained on imagenet")
flags.DEFINE_string("source_name", "train_dslr", "the name of source images")
flags.DEFINE_string("target_name", "train_amazon", "the name of target images")
flags.DEFINE_string("checkpoint_name", "resnet_v1_50.ckpt", "the name of checkpoint")

flags.DEFINE_integer("amazon_size", 2817, "the size of amazon dataset")
flags.DEFINE_integer("dslr_size", 498, "the size of amazon dataset")
flags.DEFINE_integer("webcam_size", 795, "the size of amazon dataset")
FLAGS = flags.FLAGS


def main(_):
    model_1 = DJARTN(FLAGS)
    result = []
    best_result = 0
    best_step = 0
    model_1.train_model()
    for step in range(225, 226):
        precision = model_1.test_model(step * 100)
        result.append(precision)
        if precision > best_result:
            best_result = precision
            best_step = step
        print('step:', step)
        print('precision:', precision)
        print('best_result:', best_result)
        print("best_step:", best_step)

    sio.savemat('result.mat', {'prec': result})


if __name__ == '__main__':
    tf.app.run()




