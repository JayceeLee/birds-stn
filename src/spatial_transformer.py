import numpy as np
import tensorflow as tf
import sys

sys.path.append('/home/jason/models/research/slim/')
from nets.inception_v2 import *
from tensorflow.contrib import slim

INCEPTION_CKPT = '/home/jason/models/checkpoints/inception_v2.ckpt'

def transform(U, theta, out_size, name='transform'):
    """Spatial Transformer Layer
    Implements a spatial transformer layer <https://arxiv.org/abs/1506.02025>
    Code lifted from tensorlayer.layers <https://github.com/zsdonghao/tensorlayer>

    Args:
        U : float
            The output of a convolutional net should have the
            shape [num_batch, height, width, num_channels].
        theta: float
            The output of the
            localisation network should be [num_batch, 6].
        out_size: tuple of two ints
            The size of the output of the network (out_height, out_width)

    Returns:
        output: The transformed output i.e a tensor of shape
            (num_batch, out_height, out_width, num_channels)

    """
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            x = (x + 1.0)*(width_f) / 2.0
            y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output

    def _meshgrid(height, width):
        with tf.variable_scope('_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                                tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                                tf.ones(shape=tf.stack([1, width])))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        with tf.variable_scope('_transform'):
            print('tf.shape(imput_dim):', input_dim.get_shape())
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = input_dim.get_shape()[3] # channels is static, must be known in advance
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')

            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]

            #print('Transform num_batch: %s'%(input_dim.get_shape()[0]))
            #print('Transform height: %s'%(input_dim.get_shape()[1]))
            #print('Transform width: %s'%(input_dim.get_shape()[2]))
            #print('Transform num_channels: %s'%(input_dim.get_shape()[3]))
            #print('Transform out_height: %d'%(out_size[0]))
            #print('Transform out_width: %d'%(out_size[1]))

            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))

            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])

            input_transformed = _interpolate(
                    input_dim, x_s_flat, y_s_flat,
                    out_size)

            output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            print('output shape:', output.get_shape())
            print('num channels:', num_channels)
            return output

    print('Transform input: %s'%(U.get_shape()))
    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output

class Localizer:
    """Implements 6DOF fuly-connected localizer network with num_keysx6 output layer
    """
    def __init__(self, num_keys,reuse=False, name='localizer'):
        self.name=name
        self.reuse=reuse
        #tf.constant(0.3, dtype=tf.float32, name='keep_prob')
        self.keep_prob = 0.3
        self.num_keys= num_keys
        self.theta_dim=6
        self.fc_loc1_units=50

    def localize(self,x):
        """Creates a localization network (2 FC layers) to estimate 
        the 6DOF localization parameters
        Args:
            x: A tensor of shape (batch_size, n_features)

        Returns:
            h_fc_loc2: A tensor of shape (batch_size, 6xnum_keys) 

        """
        print('Localize input x: %s'%(x.get_shape()))
        with tf.variable_scope('localize',reuse=self.reuse):
            # use MLP as the localisation net
            W_fc_loc1 = tf.get_variable(name='W_fc_loc1', initializer=tf.zeros([
                x.get_shape()[1],self.fc_loc1_units]))
            b_fc_loc1 = tf.get_variable(name='b_fc_loc1', initializer=tf.random_normal([
                self.fc_loc1_units], mean=0.0, stddev=0.01))

            W_fc_loc2 = tf.get_variable(name='W_fc_loc2', initializer=tf.zeros([
                self.fc_loc1_units,self.theta_dim*self.num_keys]))
            # Use identity transformation as starting point
            identity = np.array([[1., 0, 0], [0, 1., 0]]) #dim should match self.theta_dim*self.num_keys
            initial = np.tile(identity,[self.num_keys,1])
            initial = initial.astype('float32')
            initial = initial.flatten()
            b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

            # %% Define the two layer localisation network
            h_fc_loc1 = tf.nn.tanh(tf.matmul(x, W_fc_loc1) + b_fc_loc1)
            # %% We can add dropout for regularizing and to reduce overfitting:
            h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, self.keep_prob)
            # %% Second layer
            h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, W_fc_loc2) + b_fc_loc2)

            return h_fc_loc2


class LocalizerRNN:
    """Implements 6DOF fuly-connected localizer network with 'theta_dim' output layer
    """
    def __init__(self, num_keys, batch_size, reuse=False, name='localizer'):
        self.name=name
        self.reuse=reuse
        self.keep_prob = 0.3
        self.num_keys= num_keys
        self.theta_dim=6
        self.fc_loc1_units=50
        self.rnn_size=200
        self.batch_size=batch_size

        with tf.variable_scope('localize',reuse=self.reuse):
            self.RNN = tf.nn.rnn_cell.BasicLSTMCell(self.rnn_size)
            # use MLP to get localization parameters
            self.W_fc_loc1 = tf.get_variable(name='W_fc_loc1', initializer=tf.zeros([
                self.rnn_size,self.fc_loc1_units]))
            self.b_fc_loc1 = tf.get_variable(name='b_fc_loc1', initializer=tf.random_normal([
                self.fc_loc1_units], mean=0.0, stddev=0.01))

            self.W_fc_loc2 = tf.get_variable(name='W_fc_loc2', initializer=tf.zeros([
                self.fc_loc1_units,self.theta_dim]))
            # Use identity transformation as starting point
            initial = np.array([[1., 0, 0], [0, 1., 0]]) #dim should match self.theta_dim*self.num_keys
            #initial = np.tile(identity,[self.num_keys,1])
            initial = initial.astype('float32')
            initial = initial.flatten()
            self.b_fc_loc2 = tf.Variable(initial_value=initial, name='b_fc_loc2')

    def reset_state(self):
        self.prev_state=self.RNN.zero_state(self.batch_size, tf.float32)

    def localize(self,x):
        """Creates a localization network (2 FC layers) to estimate 
        the 6DOF localization parameters
        Args:
            x: A tensor of shape (batch_size, n_features)

        Returns:
            h_fc_loc2: A tensor of shape (batch_size, theta_dim) 

        """
        print('Localize input x: %s'%(x.get_shape()))
        with tf.variable_scope('localize',reuse=self.reuse):
            RNN_output, RNN_state = self.RNN(x, self.prev_state)
            print('RNN_output shape: %s'%(RNN_output.get_shape()))

            # %% Define the two layer localisation network
            h_fc_loc1 = tf.nn.tanh(tf.matmul(RNN_output, self.W_fc_loc1) + self.b_fc_loc1)
            # %% We can add dropout for regularizing and to reduce overfitting:
            h_fc_loc1_drop = tf.nn.dropout(h_fc_loc1, self.keep_prob)
            # %% Second layer
            h_fc_loc2 = tf.nn.tanh(tf.matmul(h_fc_loc1_drop, self.W_fc_loc2) + self.b_fc_loc2)
            print('h_fc_loc2 shape: %s'%(h_fc_loc2.get_shape()))

            self.prev_state = RNN_state

            return h_fc_loc2


class LocalizerInception(object):
    """ Implements a 3DOF inception base localizer network with theta_dim output layer
    """

    def __init__(self, num_keys, theta_dim, batch_size, reuse=False, name='localizer'):
        self.theta_dim = theta_dim # number of transform parameters (x, y)
        self.num_keys  = num_keys # number of attention glimpes
        self.batch_size = batch_size
        self.name = name
        self.reuse = reuse

    def localize(self, x, is_training):
        """Creates an inception localization network with added FC final layer to estimate transform params
        Args:
            x: A tensor of shape (batch_size, n_features)

        Returns:
            h_fc_loc2: A tensor of shape (batch_size, theta_dimxnum_keys)

        """
        # TODO: this is going to cause a problem when finding variables
        with tf.variable_scope('localize', reuse=self.reuse):
            with tf.contrib.framework.arg_scope(inception_v2_arg_scope()):
                logits, end_points = inception_v2(x, num_classes=None, is_training=is_training)
            with tf.variable_scope('added_layers'):
                # 8 x 8 x 1024
                inception_features = end_points['Mixed_5b']
                print('localize feats:', inception_features.get_shape())
                # TODO: not sure if there should be activations between layers
                # 1 x 1 conv with 128 out channels
                conv = tf.layers.conv2d(inputs=inception_features, filters=128, kernel_size=[1, 1], padding="same", activation=tf.nn.relu, name='added_1x1conv')
                print('localize conv:', conv.get_shape())
                # FC layer with 128 dim output (8x8x128 -> 128)
                conv_flat = tf.reshape(conv, [self.batch_size, -1])
                fc   = tf.layers.dense(inputs=conv_flat, units=128, activation=tf.nn.relu, name='added_fc')
                print('localize fc:', fc.get_shape())
                # FC layer that outputs theta params
                out  = tf.layers.dense(inputs=fc, units=self.theta_dim*self.num_keys, name='added_out')
                print('localize out:', out.get_shape())
                return out


if __name__ == '__main__':
    # TODO: test Inception Localizer
    x   = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
    net = LocalizerInception(2, 2)
    out = net.localize(x, is_training=True)
    init = tf.global_variables_initializer()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    cnn_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='localize/InceptionV2')
    print('cnn vars:', cnn_vars)
    # map names in checkpoint to variables to init
    cnn_vars = {v.name.split('localize/')[1][0:-2] : v  for v in cnn_vars}
    print('cnn_vars:', cnn_vars)
    #sess.run(init)
    assign_fn = tf.contrib.framework.assign_from_checkpoint_fn(INCEPTION_CKPT, cnn_vars, ignore_missing_vars=False, reshape_variables=False)
    assign_fn(sess)


