import tensorflow as tf
from tensorflow.python import debug as tf_debug
import argparse
import copy
import numpy as np
import os
import sys
from datetime import datetime
from utils.get_data import CUBDataLoader

sys.path.append('/home/jason/models/research/slim/')
from nets.inception_v3 import *
from tensorflow.contrib import slim

INCEPTION_CKPT = '/home/jason/models/checkpoints/inception_v3.ckpt'

CURRENT_DIR = os.path.dirname(__file__)

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default=os.path.join('/dvmm-filer2/datasets/ImageNet/train/'))
parser.add_argument('--checkpoint_dir', type=str, default=os.path.join(CURRENT_DIR, '../checkpoints'))
parser.add_argument('--output_dir', type=str, default=os.path.join(CURRENT_DIR, '../out'))
parser.add_argument('--log_dir', type=str, default=os.path.join(CURRENT_DIR, '../logs'))
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--unpause', action='store_true', default=False)

parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate step-size')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_max_epochs', type=int, default=1000)
parser.add_argument('--image_dim', type=int, default=(224, 224, 3))
parser.add_argument('--num_steps_per_checkpoint', type=int, default=100, help='Number of steps between checkpoints')


class CNN(object):

    def __init__(self, config):
        date_time = datetime.today().strftime('%Y%m%d_%H%M%S')
        data = 'cubirds'
        hyper_params = 'lr_'+str(config.lr)+'_max_ep_'+str(config.num_max_epochs)+'_data_'+str(data)
        subdir = date_time + '_' + hyper_params

        self.log_dir = config.log_dir + '/' + subdir
        self.checkpoint_dir = config.checkpoint_dir + '/' + subdir
        self.output_dir = config.output_dir + '/' + subdir
        self.data_dir = config.data_dir
        self.checkpoint = config.checkpoint

        self.lr = config.lr
        self.batch_size = config.batch_size
        self.num_max_epochs = config.num_max_epochs
        self.image_dim = config.image_dim

        self.num_steps_per_checkpoint = config.num_steps_per_checkpoint
        self.config = copy.deepcopy(config)

        # set up model
        self.load_data()
        self.add_placeholders()
        self.logits = self.add_model(self.x_placeholder, self.is_training_placeholder)
        self.preds, self.accuracy_op = self.predict(self.logits, self.y_placeholder)
        self.loss, self.loss_summary = self.add_loss_op(self.logits, self.y_placeholder)
        self.train_op = self.add_training_op(self.loss)

    def load_data(self):
        cub = CUBDataLoader()
        train_test_files = [os.path.join(CURRENT_DIR, 'utils/train.txt'), os.path.join(CURRENT_DIR, 'utils/test.txt')]
        (train_x, train_y), (test_x, test_y) = cub.get_data(train_test_files)
        train_x = tf.constant(train_x)
        train_y = tf.constant(train_y)
        test_x  = tf.constant(test_x)
        test_y  = tf.constant(test_y)

        train_data = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        train_data = train_data.map(self.preprocess_image).batch(self.batch_size)
        test_data  = tf.data.Dataset.from_tensor_slices((test_x, test_y))
        test_data  = test_data.map(self.preprocess_image).batch(self.batch_size)

        self.iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
        self.next_batch = self.iterator.get_next()

        self.train_init_op = self.iterator.make_initializer(train_data)
        self.test_init_op  = self.iterator.make_initializer(test_data)

    def preprocess_image(self, image_path, y):
        image_file    = tf.read_file(image_path)
        image_decoded = tf.image.decode_jpeg(image_file, channels=3)
        image_resized = tf.image.resize_images(image_decoded, [224, 224])
        return image_resized, y

    def add_placeholders(self):
        height, width, channels = self.image_dim
        with tf.name_scope('data'):
            self.x_placeholder = tf.placeholder(tf.float32, shape=(None, height, width, channels), name='features')
            self.y_placeholder = tf.placeholder(tf.int32, shape=None, name='labels')
            self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

    def create_feed_dict(self, x, y, is_training):
        feed_dict = {
            self.x_placeholder : x,
            self.y_placeholder : y,
            self.is_training_placeholder : is_training
        }
        return feed_dict

    def add_model(self, images, is_training):
        with tf.contrib.framework.arg_scope(inception_v3_arg_scope()):
            logits, end_points = inception_v3(images, num_classes=None, is_training=True)
        feats = end_points['AvgPool_1a']
        flat_feats = tf.squeeze(feats)
        fc_weights = tf.get_variable('fc-weights', [2048, 200], initializer=tf.random_normal_initializer())
        logits = tf.matmul(flat_feats, fc_weights, name='fc')
        return logits

    def add_training_op(self, loss):
        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.lr)
            #train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'InceptionV3/AuxLogits')
            #train_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'InceptionV3/Logits')
            train_vars = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if 'Inception' not in v.name]
            print('train_vars:', train_vars)
            self.global_step = tf.train.get_or_create_global_step()
            train_op = optimizer.minimize(loss, self.global_step, var_list=train_vars)
        return train_op

    def add_loss_op(self, logits, y):
        with tf.name_scope('loss'):
            entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
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
            with tf.name_scope('accuracy'):
                correct_preds = tf.equal(tf.cast(predictions, tf.int32), y)
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
                    _, loss, summary, step = sess.run([self.train_op, self.loss, self.loss_summary, self.global_step], feed_dict=feed_dict)
                    ave_loss += loss
                    count    += 1.0
                    self.summary_writer.add_summary(summary, step)
                    losses.append(loss)
                except tf.errors.OutOfRangeError:
                   print('average loss:', ave_loss/count, 'epoch:', i)
                   break

            # save every 10 epochs
            if (i + 1) % 10 == 0:
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
    saver = tf.train.Saver()
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    with tf.control_dependencies(update_ops):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # if debug mode on
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess)
        #sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

        checkpoint = tf.train.get_checkpoint_state(os.path.dirname(net.checkpoint_dir))
        if net.checkpoint != None:
            print 'restoring from checkpoint:', net.checkpoint
            checkpoint_path = net.checkpoint
        elif checkpoint and checkpoint.model_checkpoint_path and args.unpause:
            print 'restoring from checkpoint:', checkpoint.model_checkpoint_path
            net.checkpoint = checkpoint.model_checkpoint_path
            checkpoint_path = checkpoint.model_checkpoint_path
        else:
            print 'training from scratch'
            cnn_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='InceptionV3')
            cnn_vars = [v for v in cnn_vars if 'Adam' not in v.name] # don't restore optim vars
            checkpoint_path = INCEPTION_CKPT
            print('cnn vars:', cnn_vars)
            saver = tf.train.Saver(var_list=cnn_vars)#, max_to_keep=5)
            sess.run(init)
        print('checkpoint path:', checkpoint_path)
        saver.restore(sess, checkpoint_path)
        losses = net.fit(sess, saver)
