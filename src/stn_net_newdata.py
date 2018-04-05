import tensorflow as tf
from tensorflow.python import debug as tf_debug
import argparse
import copy
import numpy as np
import os
import glob
import sys
from datetime import datetime

# from tf_classification
from tf_classification.preprocessing.decode_example import decode_serialized_example

# from models repo
sys.path.append('/home/jason/models/research/slim/')
from nets.inception_v3 import inception_v3, inception_v3_arg_scope
from tensorflow.contrib import slim

# from me
from utils.get_data import CUBDataLoader
from spatial_transformer import *

# from models repo
sys.path.append('/home/jason/models/research/slim/')
from nets.inception_v3 import *
from tensorflow.contrib import slim
#from preprocessing.inception_preprocessing import *

# pre-trained imagenet inception v3 model checkpoint
INCEPTION_CKPT = '/home/jason/models/checkpoints/inception_v3.ckpt'

CURRENT_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
# data_dir/train* or data_dir/test* or data_dir/val*
parser.add_argument('--data_dir', type=str, default=os.path.join('/dvmm-filer2/users/jason/tensorflow_datasets/cub/with_600_val_split/'))
parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(CURRENT_DIR, '../checkpoints'))
parser.add_argument('--output_dir', type=str, default=os.path.join(CURRENT_DIR, '../out'))
parser.add_argument('--log_dir', type=str, default=os.path.join(CURRENT_DIR, '../logs'))
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--unpause', action='store_true', default=False)
parser.add_argument('--save', action='store_true', default=True)
parser.add_argument('--seed', type=float, default=1, help='Random seed')
parser.add_argument('--lr', type=float, default=0.1, help='Learning rate step-size')
parser.add_argument('--mode', type=str, default='test', help='train or test mode')
#parser.add_argument('--stn_lr', type=float, default=0.001, help='Learning rate step-size for STN')
parser.add_argument('--batch_size', type=int, default=32) # TODO: see if you can fit 256
parser.add_argument('--num_max_iters', type=int, default=300000)
parser.add_argument('--image_dim', type=int, default=(244, 244, 3)) # TODO: change to 448
parser.add_argument('--num_steps_per_checkpoint', type=int, default=100, help='Number of steps between checkpoints')
parser.add_argument('--num_crops', type=int, default=2, help='Number of attention glimpses over input image')
parser.add_argument('--image_processing_train', type=str, default='/home/jason/birds-stn/src/cub_image_config_train.yaml', help='Path to the image pre-processing config file')
parser.add_argument('--image_processing_test', type=str, default='/home/jason/birds-stn/src/cub_image_config_test.yaml', help='Path to the image pre-processing config file')


class CNN(object):
    def __init__(self, config):
        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        data = 'cubirds'
        hyper_params = 'lr_'+str(config.lr)+'_max_iters_'+str(config.num_max_iters)+'_data_'+str(data)
        subdir = date_time + '_mode_'+config.mode+'_stn_unfroze_imagenet_v3_aux_logits_' + hyper_params # this version uses inception v3 unfrozen

        checkpoint_state = tf.train.get_checkpoint_state(config.checkpoint_dir)
        self.checkpoint = checkpoint_state.model_checkpoint_path if checkpoint_state else config.checkpoint
        self.save = config.save

        self.log_dir = config.log_dir + '/' + config.mode + '/' + subdir
        self.checkpoint_dir = config.checkpoint_dir + '/' + subdir
        self.output_dir = config.output_dir + '/' + subdir
        self.data_dir = config.data_dir
        self.train_preprocessing_config = config.image_processing_train
        self.test_preprocessing_config = config.image_processing_test
        self.mode = config.mode

        self.lr = config.lr
        self.batch_size = config.batch_size
        self.num_max_iters = config.num_max_iters
        self.image_dim = config.image_dim
        self.num_crops = config.num_crops
        self.num_classes = 200
        self.seed = config.seed
        tf.set_random_seed(self.seed)

        self.num_steps_per_checkpoint = config.num_steps_per_checkpoint
        self.config = copy.deepcopy(config)

        # set up model
        self.add_placeholders()
        self.load_data()
        self.localizer = LocalizerInceptionV3(num_keys=self.num_crops, theta_dim=2, batch_size=self.batch_size)
        self.logits, self.aux_logits = self.add_model(self.x_placeholder, self.is_training_placeholder)
        self.preds, self.accuracy_op = self.predict(self.logits, self.y_placeholder)
        self.loss, self.loss_summary = self.add_loss_op(self.logits, self.aux_logits, self.y_placeholder)
        self.train_op = self.add_training_op(self.loss)

    def load_data(self):
        train_path = os.path.join(self.data_dir, 'train*')
        train_data = self.batched_dataset_from_records(train_path, mode='train', batch_size=self.batch_size)
        train_iter = train_data.make_one_shot_iterator()

        val_path = os.path.join(self.data_dir, 'val*')
        val_data = self.batched_dataset_from_records(val_path, mode='train', batch_size=self.batch_size)
        val_iter = val_data.make_one_shot_iterator()

        test_path = os.path.join(self.data_dir, 'test*')
        test_data = self.batched_dataset_from_records(test_path, mode='test', batch_size=self.batch_size)
        test_iter = test_data.make_one_shot_iterator()

        # to switch between train and test, set the handle_placeholder value to train_handle or test_handle
        iterator = tf.data.Iterator.from_string_handle(self.handle_placeholder, train_data.output_types, train_data.output_shapes)
        self.next_batch = iterator.get_next()
        self.train_handle = train_iter.string_handle()
        self.val_handle = val_iter.string_handle()
        self.test_handle = test_iter.string_handle()

    def batched_dataset_from_records(self, records_path, mode='train', batch_size=32):
        files = tf.data.Dataset.list_files(records_path)
        # parallelize creation reading of records
        #files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=4))
        dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=4)
        # suffle dataset so batches don't contain single class etc, then repeat for epochs
        #dataset = tf.contrib.data.shuffle_and_repeat(dataset)
        dataset = dataset.shuffle(buffer_size=6000) # next item randomly chosen from buffer of buffer_size
        dataset = dataset.repeat()
        # parse serialized examples in parallel
        dataset = dataset.map(lambda ex : self.parser(ex, mode), num_parallel_calls=8)
        # create batches
        dataset = dataset.batch(batch_size)
        # allow generation of data and consumption of data to occur at the same time
        dataset = dataset.prefetch(buffer_size=batch_size)
        return dataset

    def parser(self, serialized_example, mode='train'):
        features_to_fetch = [
            ('image/encoded', 'image'), ('image/class/label', 'label'),
            ('image/height', 'height'), ('image/width', 'width'),
            ('image/channels', 'channels')
        ]
        # extract tensor values from serialized example
        example_dict = decode_serialized_example(serialized_example, features_to_fetch)
        label = example_dict['label']
        image = example_dict['image']
        # set image size
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        height, width, channels = example_dict['height'], example_dict['width'], example_dict['channels']
        height, width, channels = tf.cast(height, tf.int32), tf.cast(width, tf.int32), tf.cast(channels, tf.int32)
        image = tf.reshape(image, ([height, width, channels]))
        new_height, new_width, new_channels = self.image_dim
        if mode == 'train':
            # resize the image to 256xS where S is max(largest-image-side, 244)
            image = tf.expand_dims(image, 0)
            clipped_height, clipped_width = tf.maximum(height, [244]), tf.maximum(width, [244])
            true_fn = lambda : tf.image.resize_bilinear(image, [clipped_height[0], 256], align_corners=False)
            false_fn = lambda : tf.image.resize_bilinear(image, [256, clipped_width[0]], align_corners=False)
            image = tf.cond(tf.greater(height, width), true_fn, false_fn)
            image = tf.squeeze(image, [0])
            # preprocess with random crops and horizontal flipping
            image = tf.random_crop(image, size=[new_height, new_width, new_channels])
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.central_crop(image, central_fraction=0.875)
            image = tf.expand_dims(image, 0)
            image = tf.image.resize_bilinear(image, [new_height, new_width])
            image = tf.squeeze(image, [0])
        return image, tf.one_hot(label, self.num_classes)

    def add_placeholders(self):
        height, width, channels = self.image_dim
        with tf.name_scope('data'):
            self.x_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, height, width, channels], name='images')
            self.y_placeholder = tf.placeholder(tf.int32, shape=[self.batch_size, self.num_classes], name='labels')
            self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')
            self.handle_placeholder = tf.placeholder(tf.string, shape=[], name='handle')

    def create_feed_dict(self, x, y, is_training, handle):
        feed_dict = {
            self.x_placeholder : x,
            self.y_placeholder : y,
            self.is_training_placeholder : is_training,
            self.handle_placeholder : handle
        }
        return feed_dict

    def add_model(self, images, is_training):
        # get predicated theta values
        #tf.summary.image("original", images, self.batch_size, collections=None)
        # contains n*2 params for n transforms
        theta = self.localizer.localize(images, is_training)
        # print theta's over time
        self.theta = theta
        theta_list = tf.split(theta, num_or_size_splits=self.num_crops, axis=1)
        transform_list = []
        # transform the images using theta
        for i in range(len(theta_list)):
            theta_i = tf.reshape(theta_list[i], [-1, 2, 1])
            tf.summary.histogram('histogram_theta_'+str(i), theta_list[i])
            # add the fixed size scale transform parameters
            theta_scale = tf.eye(2, batch_shape=[self.batch_size]) * 0.5
            theta_i = tf.concat([theta_scale, theta_i], axis=2)
            # flatten thetas for transform
            theta_i = tf.reshape(theta_i, [self.batch_size, 6])
            transform_i = transform(images, theta_i, out_size=(224, 224))
            transform_list.append(transform_i)
            #tf.summary.image('transform_'+str(i), transform_i, self.batch_size, collections=None)
        # extract features
        features_list = []
        aux_logits_list = []
        with tf.variable_scope('classifier'):
            with tf.contrib.framework.arg_scope(inception_v3_arg_scope()):
                for i in range(len(transform_list)):
                    reuse = True if i > 0 else False
                    transform_i = transform_list[i]
                    _, end_points_i = inception_v3(transform_i, num_classes=self.num_classes, is_training=is_training, reuse=reuse)
                    # TODO: check if this should be changed to something other than AbgPool_1a
                    aux_logits_i = end_points_i['AuxLogits']
                    aux_logits_list.append(aux_logits_i)
                    features_i = tf.squeeze(end_points_i['AvgPool_1a'], axis=[1,2], name='feats'+str(i))
                    features_list.append(features_i)
            features = tf.concat(features_list, axis=1)
            dropout = tf.nn.dropout(features, 0.7)
            with tf.variable_scope('final_out'):
                logits = tf.layers.dense(dropout, self.num_classes, name='feats2out')
        return logits, aux_logits_list

    def add_training_op(self, loss):
        with tf.name_scope('optimizer'):
            self.global_step = tf.train.get_or_create_global_step()
            optimizer_out = tf.train.GradientDescentOptimizer(self.lr)
            out_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
            optimizer_localizer = tf.train.GradientDescentOptimizer(self.lr * 1e-4)
            localizer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='localize')
            # add update ops for batch norm, note this causes crash if done in main
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op_out = optimizer_out.minimize(loss, self.global_step, var_list=out_vars)
                train_op_localizer = optimizer_localizer.minimize(loss, self.global_step, var_list=localizer_vars)
                train_op = tf.group(train_op_out, train_op_localizer)
        return train_op

    def add_loss_op(self, logits, aux_logits, y):
        with tf.name_scope('loss'):
            aux_entropy_list = [tf.losses.softmax_cross_entropy(onehot_labels=y, logits=x, weights=0.4, reduction=tf.losses.Reduction.NONE) \
                                for x in aux_logits]
            aux_entropy = tf.reduce_sum(aux_entropy_list, axis=0)
            entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, weights=1.0, reduction=tf.losses.Reduction.NONE)
            entropy += aux_entropy
            loss = tf.reduce_mean(entropy, name='loss')
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', loss)
            tf.summary.histogram('histogram loss', loss)
            summary_op = tf.summary.merge_all()
        return loss, summary_op

    def predict(self, logits, y=None):
        """Make predictions from the provided model.
        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          average_loss: Average loss of model.
          predictions: Predictions of model on input_data
        """
        with tf.name_scope('predictions'):
            predictions = tf.argmax(tf.nn.softmax(logits), 1)
        if y != None:
            labels = tf.argmax(y, 1)
            with tf.name_scope('accuracy'):
                correct_preds = tf.equal(tf.cast(predictions, tf.int32), tf.cast(labels, tf.int32))
                accuracy_op = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
        return predictions, accuracy_op

    def fit(self, sess, saver):
        """Fit model on provided data.

        Args:
          sess: tf.Session()
          input_data: np.ndarray of shape (n_samples, n_features)
          input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
          losses: list of loss per epoch
        """
        losses = []
        logdir = self.log_dir
        self.summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        self.train_handle = sess.run(self.train_handle)
        self.test_handle = sess.run(self.test_handle)

        for i in range(self.num_max_iters):
            batch_x, batch_y = sess.run(self.next_batch, feed_dict={self.handle_placeholder : self.train_handle})
            feed_dict = self.create_feed_dict(batch_x, batch_y, True, self.train_handle)
            _, loss, summary, step, theta, accuracy = sess.run([self.train_op, self.loss, self.loss_summary, self.global_step, self.theta, self.accuracy_op], feed_dict=feed_dict)
            print('iter i:', i, 'loss:', loss, 'accuracy:', accuracy)
            self.summary_writer.add_summary(summary, step)
            losses.append(loss)

            # learning rate schedule 10k, 20k, and 25k
            if (i + 1) % 10000 == 0 or (i + 1) % 20000 == 0 or (i + 1) % 25000 == 0:
                self.lr = self.lr * 0.1

            # save checkpoint
            if (i + 1) % 500 == 0 and self.save:
                saver.save(sess, self.checkpoint_dir, step)

            # evaluate on 100 batches of test set
            if (i + 1) % 500 == 0:
                ave_test_accuracy = 0.0
                for j in range(100):
                    batch_x, batch_y = sess.run(self.next_batch, feed_dict={self.handle_placeholder : self.test_handle})
                    feed_dict = self.create_feed_dict(batch_x, batch_y, False, self.test_handle)
                    accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
                    ave_test_accuracy += accuracy
                ave_test_accuracy = ave_test_accuracy / 100.0
                print('average test accuracy:', ave_test_accuracy, 'epoch:', i)
                summary = tf.Summary()
                summary.value.add(tag='100 mini_batch test accuracy', simple_value=ave_test_accuracy)
                self.summary_writer.add_summary(summary, step)

            # evaluate on 100 batches of train set
            if (i + 1) % 500 == 0:
                ave_train_accuracy = 0.0
                for j in range(100):
                    batch_x, batch_y = sess.run(self.next_batch, feed_dict={self.handle_placeholder : self.train_handle})
                    feed_dict = self.create_feed_dict(batch_x, batch_y, False, self.train_handle)
                    accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
                    ave_train_accuracy += accuracy
                ave_train_accuracy = ave_train_accuracy / 100.0
                print('average train accuracy:', ave_train_accuracy, 'epoch:', i)
                summary = tf.Summary()
                summary.value.add(tag='100 mini_batch train accuracy', simple_value=ave_train_accuracy)
                self.summary_writer.add_summary(summary, step)
        return losses

if __name__ == "__main__":
    args = parser.parse_args()
    net = CNN(args)
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # initalize localization vars to inception pre-trained weights
    localizer_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='localize/InceptionV3')
    # map names in checkpoint to variables to init
    localizer_vars = {v.name.split('localize/')[1][0:-2] : v  for v in localizer_vars}
    cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='InceptionV3')
    cnn_vars = [v for v in cnn_vars if 'Adam' not in v.name]
    cnn_vars = [v for v in cnn_vars if 'BatchNorm' not in v.name]
    cnn_vars = {v.name[0:-2] : v for v in cnn_vars}
    # combine dictionaries
    cnn_vars.update(localizer_vars)
    #print('cnn_vars:', cnn_vars)
    saver = tf.train.Saver(max_to_keep=3)
    vars_to_save = slim.get_model_variables().append(net.global_step)
    sess = tf.Session(vars_to_save, config=config)
    sess.run(init)
    assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(INCEPTION_CKPT, cnn_vars, ignore_missing_vars=True, reshape_variables=False)
    assign_fn(sess)
    losses = net.fit(sess, saver)
