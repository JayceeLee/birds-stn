import tensorflow as tf
from tensorflow.python import debug as tf_debug
import argparse
import copy
import numpy as np
import os
import sys
from datetime import datetime

from utils.get_data import CUBDataLoader
from spatial_transformer import *

sys.path.append('/home/jason/models/research/slim/')
from nets.inception_v3 import *
from tensorflow.contrib import slim
from preprocessing.inception_preprocessing import *

# pre-trained imagenet and cub inception v3 model checkpoint
INCEPTION_CKPT = '/home/jason/temp/cub_image_experiment/logdir/'

CURRENT_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.join('/dvmm-filer2/datasets/ImageNet/train/'))
parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(CURRENT_DIR, '../checkpoints'))
parser.add_argument('--output_dir', type=str, default=os.path.join(CURRENT_DIR, '../out'))
parser.add_argument('--log_dir', type=str, default=os.path.join(CURRENT_DIR, '../logs'))
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--unpause', action='store_true', default=False)
parser.add_argument('--save', action='store_true', default=True)

parser.add_argument('--lr', type=float, default=0.1, help='Learning rate step-size')
#parser.add_argument('--stn_lr', type=float, default=0.001, help='Learning rate step-size for STN')
parser.add_argument('--batch_size', type=int, default=32) # TODO: see if you can fit 256
parser.add_argument('--num_max_epochs', type=int, default=1500)
parser.add_argument('--image_dim', type=int, default=(244, 244, 3)) # TODO: change to 448
parser.add_argument('--num_steps_per_checkpoint', type=int, default=100, help='Number of steps between checkpoints')
parser.add_argument('--num_crops', type=int, default=2, help='Number of attention glimpses over input image')


class CNN(object):
    def __init__(self, config):
        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        data = 'cubirds'
        hyper_params = 'lr_'+str(config.lr)+'_max_ep_'+str(config.num_max_epochs)+'_data_'+str(data)
        subdir = date_time + '_stn_unfroze_v3_' + hyper_params # this version uses inception v3 unfrozen

        self.log_dir = config.log_dir + '/' + subdir
        self.checkpoint_dir = config.checkpoint_dir + '/' + subdir
        self.output_dir = config.output_dir + '/' + subdir
        self.data_dir = config.data_dir
        self.checkpoint = config.checkpoint
        self.save = config.save

        self.lr = config.lr
        #self.stn_lr = config.stn_lr
        self.batch_size = config.batch_size
        self.num_max_epochs = config.num_max_epochs
        self.image_dim = config.image_dim
        self.num_crops = config.num_crops

        self.num_steps_per_checkpoint = config.num_steps_per_checkpoint
        self.config = copy.deepcopy(config)

        # set up model
        self.load_data()
        self.add_placeholders()
        self.localizer = LocalizerInceptionV3(num_keys=self.num_crops, theta_dim=2, batch_size=self.batch_size)
        self.logits = self.add_model(self.x_placeholder, self.is_training_placeholder)
        self.preds, self.accuracy_op = self.predict(self.logits, self.y_placeholder)
        self.loss, self.loss_summary = self.add_loss_op(self.logits, self.y_placeholder)
        self.train_op = self.add_training_op(self.loss)

    def load_data(self):
        cub = CUBDataLoader()
        train_test_files = [os.path.join(CURRENT_DIR, 'utils/train.txt'), os.path.join(CURRENT_DIR, 'utils/test.txt')]
        (train_x, train_y), (test_x, test_y) = cub.get_data(train_test_files)
        train_x = tf.constant(train_x)
        train_y = tf.one_hot(tf.constant(train_y), 200)
        test_x  = tf.constant(test_x)
        test_y = tf.one_hot(tf.constant(test_y), 200)

        train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_data = train_data.map(lambda x, y: self.preprocess_image(x, y, True))
        train_data = train_data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))
        test_data  = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_data  = test_data.map(lambda x, y: self.preprocess_image(x, y, False))
        test_data = test_data.apply(tf.contrib.data.batch_and_drop_remainder(self.batch_size))

        self.iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        self.next_batch = self.iterator.get_next()

        self.train_init_op = self.iterator.make_initializer(train_data)
        self.test_init_op  = self.iterator.make_initializer(test_data)

    def preprocess_image(self, image_path, y, is_training):
        height, width, channels = self.image_dim
        image_file    = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_file, channels=channels)
        image_resized = preprocess_image(image_decoded, height, width, is_training)
        image_normalized = tf.image.per_image_standardization(image_resized)
        return image_normalized, y

    def add_placeholders(self):
        height, width, channels = self.image_dim
        with tf.name_scope('data'):
            self.x_placeholder = tf.placeholder(tf.float32, shape=(self.batch_size, height, width, channels), name='features')
            self.y_placeholder = tf.placeholder(tf.int32, shape=(self.batch_size, 200), name='labels') # one hots
            self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

    def create_feed_dict(self, x, y, is_training):
        feed_dict = {
            self.x_placeholder : x,
            self.y_placeholder : y,
            self.is_training_placeholder : is_training
        }
        return feed_dict

    def add_model(self, images, is_training):
        # get predicated theta values
        tf.summary.image("original", images, self.batch_size, collections=None)
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
            tf.summary.image('transform_'+str(i), transform_i, self.batch_size, collections=None)
        # extract features
        features_list = []
        with tf.variable_scope('classifier'):
            with tf.contrib.framework.arg_scope(inception_v3_arg_scope()):
                for i in range(len(transform_list)):
                    reuse = True if i > 0 else False
                    transform_i = transform_list[i]
                    _, end_points_i = inception_v3(transform_i, num_classes=None, is_training=is_training, reuse=reuse)
                    # TODO: check if this should be changed to something other than AbgPool_1a
                    features_i = tf.squeeze(end_points_i['AvgPool_1a'], axis=[1,2], name='feats'+str(i))
                    features_list.append(features_i)
            features = tf.concat(features_list, axis=1)
            dropout = tf.nn.dropout(features, 0.7)
            with tf.variable_scope('final_out'):
                logits = tf.layers.dense(dropout, 200, name='feats2out')
        return logits

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

    def add_loss_op(self, logits, y):
        with tf.name_scope('loss'):
            entropy = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=logits, weights=1.0)
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

        for i in range(self.num_max_epochs):
            count    = 0.0
            ave_loss = 0.0
            sess.run(self.train_init_op)
            while True:
                try:
                    batch_x, batch_y = sess.run(self.next_batch)
                    feed_dict = self.create_feed_dict(batch_x, batch_y, True)
                    _, loss, summary, step, theta = sess.run([self.train_op, self.loss, self.loss_summary, self.global_step, self.theta], feed_dict=feed_dict)
                    ave_loss += loss
                    count    += 1.0
                    print('batch i:', count, 'loss:', loss, 'theta:', theta)
                    self.summary_writer.add_summary(summary, step)
                    losses.append(loss)
                except tf.errors.OutOfRangeError:
                   print('average loss:', ave_loss/count, 'epoch:', i)
                   break

            # learning rate schedule assuming ~ 20 iterations per epoch
            if (i + 1) % 500 == 0 or (i + 1) % 1000 == 0 or (i + 1) % 1250 == 0:
                self.lr = self.lr * 0.1

            # save every 10 epochs
            if (i + 1) % 10 == 0 and self.save:
                saver.save(sess, self.checkpoint_dir, step)

            # evaluate on 100 batches of test set
            sess.run(self.test_init_op)
            ave_test_accuracy = 0.0
            for j in range(100):
                batch_x, batch_y = sess.run(self.next_batch)
                feed_dict = self.create_feed_dict(batch_x, batch_y, False)
                accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
                ave_test_accuracy += accuracy
            ave_test_accuracy = ave_test_accuracy / 100.0
            print('average test accuracy:', ave_test_accuracy, 'epoch:', i)
            summary = tf.Summary()
            summary.value.add(tag='100 mini_batch test accuracy', simple_value=ave_test_accuracy)
            self.summary_writer.add_summary(summary, step)

            # evaluate on 100 batches of train set
            sess.run(self.train_init_op)
            ave_train_accuracy = 0.0
            for j in range(100):
                batch_x, batch_y = sess.run(self.next_batch)
                feed_dict = self.create_feed_dict(batch_x, batch_y, False)
                accuracy = sess.run(self.accuracy_op, feed_dict=feed_dict)
                ave_train_accuracy += accuracy
            ave_train_accuracy = ave_train_accuracy / 100.0
            print('average train accuracy:', ave_train_accuracy, 'epoch:', i)
            summary = tf.Summary()
            summary.value.add(tag='100 mini_batch train accuracy', simple_value=ave_train_accuracy)
            self.summary_writer.add_summary(summary, step)

if __name__ == "__main__":
    args = parser.parse_args()
    net = CNN(args)
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

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
    saver = tf.train.Saver(var_list=cnn_vars, max_to_keep=5)
    sess.run(init)
    ckpt = tf.train.latest_checkpoint(INCEPTION_CKPT)
    assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(ckpt, cnn_vars, ignore_missing_vars=True, reshape_variables=False)
    assign_fn(sess)
    losses = net.fit(sess, saver)
