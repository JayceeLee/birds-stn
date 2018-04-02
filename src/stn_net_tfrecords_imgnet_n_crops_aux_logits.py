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
from tf_classification.config.parse_config import parse_config_file
from tf_classification.preprocessing.inputs import input_nodes
from tf_classification.train import parse_args

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
parser.add_argument('--seed', type=float, default=1.0, help='Random seed') 
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
        self.checkpoint = config.checkpoint if config.checkpoint != None else checkpoint_state.model_checkpoint_path
        self.save = config.save

        self.log_dir = config.log_dir + '/' + config.mode + '/' + subdir
        self.checkpoint_dir = config.checkpoint_dir + '/' + subdir
        self.output_dir = config.output_dir + '/' + subdir
        self.data_dir = config.data_dir
        self.train_preprocessing_config = config.image_processing_train
        self.test_preprocessing_config = config.image_processing_test
        self.mode = config.mode

        self.lr = config.lr
        #self.stn_lr = config.stn_lr
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
        self.load_data()
        self.add_placeholders()
        self.localizer = LocalizerInceptionV3(num_keys=self.num_crops, theta_dim=2, batch_size=self.batch_size)
        self.logits, self.aux_logits = self.add_model(self.x_placeholder, self.is_training_placeholder)
        self.preds, self.accuracy_op = self.predict(self.logits, self.y_placeholder)
        self.loss, self.loss_summary = self.add_loss_op(self.logits, self.aux_logits, self.y_placeholder)
        self.train_op = self.add_training_op(self.loss)

    def load_data(self):
        # TODO: look at regularization losses
        train_cfg = parse_config_file(self.train_preprocessing_config)
        test_cfg = parse_config_file(self.test_preprocessing_config)
        with tf.device('/cpu:0'):
            if self.mode == 'train':
                train_path = os.path.join(self.data_dir, 'train*')
                train_records = glob.glob(train_path)
                self.batch_dict = input_nodes(
                    tfrecords=train_records,
                    cfg=train_cfg.IMAGE_PROCESSING,
                    num_epochs=None, # TODO: see why this is
                    batch_size=self.batch_size,
                    num_threads=6,
                    shuffle_batch=True,
                    random_seed=self.seed,
                    capacity=4000,
                    min_after_dequeue=400,  # Minimum size of the queue to ensure good shuffling
                    add_summaries=False,
                    input_type='train' # note you need ones for val, test also
                )
                self.batched_one_hot_labels = slim.one_hot_encoding(self.batch_dict['labels'], num_classes=train_cfg.NUM_CLASSES)
            elif self.mode == 'test':
                tf.logging.set_verbosity(tf.logging.DEBUG)
                test_path = os.path.join(self.data_dir, 'test*')
                test_records = glob.glob(test_path)
                print('test records:', test_records)
                self.batch_dict = input_nodes(
                    tfrecords=test_records,
                    cfg=test_cfg.IMAGE_PROCESSING,
                    num_epochs=1,
                    batch_size=test_cfg.BATCH_SIZE,
                    num_threads=test_cfg.NUM_INPUT_THREADS,
                    shuffle_batch =test_cfg.SHUFFLE_QUEUE,
                    random_seed=test_cfg.RANDOM_SEED,
                    capacity=test_cfg.QUEUE_CAPACITY,
                    min_after_dequeue=test_cfg.QUEUE_MIN,
                    add_summaries=False,
                    input_type='test'
                )
                self.batched_one_hot_labels = slim.one_hot_encoding(self.batch_dict['labels'], num_classes=test_cfg.NUM_CLASSES)
                # so that we evaluate the whole dataset once and report average accuracy
                self.num_max_iters = int(np.floor(test_cfg.NUM_TEST_EXAMPLES / float(self.batch_size)))

    def add_placeholders(self):
        height, width, channels = self.image_dim
        with tf.name_scope('data'):
            self.x_placeholder = self.batch_dict['inputs']
            self.y_placeholder = self.batched_one_hot_labels
            self.is_training_placeholder = tf.placeholder(tf.bool, name='is_training')

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
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(self.num_max_iters):
            # run training op
            feed_dict = {self.is_training_placeholder : self.mode == 'train'}
            _, loss, summary, step, accuracy, theta = sess.run([self.train_op, self.loss, self.loss_summary, self.global_step, self.accuracy_op, self.theta], feed_dict=feed_dict)
            print('iter:', i, 'loss:', loss, 'accuracy:', accuracy)
            self.summary_writer.add_summary(summary, step)
            losses.append(loss)

            # learning rate schedule 10k, 20k, and 25k
            if (i + 1) % 10000 == 0 or (i + 1) % 20000 == 0 or (i + 1) % 25000 == 0:
                self.lr = self.lr * 0.1

            if (i + 1) % 1000 == 0 and self.save:
                saver.save(sess, self.checkpoint_dir, step)
        return losses


    def test(self, sess_config, saver):
        feed_dict = {self.is_training_placeholder : self.mode == 'train'}
        variables_to_restore = slim.get_variables_to_restore(include=['classifier', 'localize'])
        labels = self.batch_dict['labels']
        metric_map = {
            'Accuracy' : slim.metrics.streaming_accuracy(labels=labels, predictions=self.preds),
            'Loss' : slim.metrics.streaming_mean(self.loss)
        }
        names_to_values, names_to_updates = slim.metrics.aggregate_metric_map(metric_map)
        slim.evaluation.evaluation_loop(
            master='',
            checkpoint_dir=self.checkpoint, # path to most recent checkpoint
            logdir=self.log_dir,
            num_evals=self.num_max_iters,
            initial_op=None,
            initial_op_feed_dict=None,
            eval_op=names_to_updates.values(),
            eval_op_feed_dict=feed_dict,
            final_op=None,
            final_op_feed_dict=None,
            summary_op=tf.summary.merge_all(),
            summary_op_feed_dict=None,
            variables_to_restore=variables_to_restore,
            eval_interval_secs=60,
            max_number_of_evaluations=None,
            session_config=sess_config,
            timeout=None
        )
        '''
        logdir = self.log_dir
        #self.summary_writer = tf.summary.FileWriter(logdir, graph=tf.get_default_graph())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        ave_accuracy = 0.0
        for i in range(self.num_max_iters):
            feed_dict = {self.is_training_placeholder : self.mode == 'train'}
            loss, summary, step, accuracy = sess.run([self.loss, self.loss_summary, self.global_step, self.accuracy_op], feed_dict=feed_dict)
            ave_accuracy += accuracy
            print('iter:', i, 'test loss:', loss, 'test accuracy:', accuracy)
            #self.summary_writer.add_summary(summary, step)
        print('ave test accuracy:', ave_accuracy / float(self.num_max_iters))
        '''


if __name__ == "__main__":
    args = parser.parse_args()
    net = CNN(args)
    init = tf.global_variables_initializer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if net.mode == 'train':
        save_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='localize')
        save_vars += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='classifier')
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
        print('cnn_vars:', cnn_vars)
        print('save vars:', save_vars)
        saver = tf.train.Saver(var_list=save_vars, max_to_keep=5)
        sess = tf.Session(config=config)
        sess.run(init)
        assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(INCEPTION_CKPT, cnn_vars, ignore_missing_vars=True, reshape_variables=False)
        assign_fn(sess)
        losses = net.fit(sess, saver)
    elif net.mode == 'test':
        saver = tf.train.Saver()
        net.test(config, saver)
        '''
        sess.run(init)
        sess.run(tf.local_variables_initializer())
        print(net.checkpoint)
        if os.path.exists(net.checkpoint):
            print('restoring from checkpoint:', net.checkpoint, ' for testing')
            saver.restore(sess, net.checkpoint)
        net.test(sess, saver)
        '''
