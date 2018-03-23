import tensorflow as tf

with tf.Graph().as_default() as graph:
    model_path = '/home/jason/models/checkpoints/inception_pb/classify_image_graph_def.pb'
    print('Model path: ', model_path)
    with gfile.FastGFile(model_path, 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      bottleneck_tensor, resized_input_tensor = (tf.import_graph_def(
          graph_def,
          name='',
          return_elements=['pool_3/_reshape:0', 'Mul:0']))
  return graph, bottleneck_tensor, resized_input_tensor












from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path

import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.slim.python.slim.nets import inception
from tensorflow.python.ops import array_ops
from tensorflow.python.training import saver as tf_saver

slim = tf.contrib.slim
FLAGS = None


def PreprocessImage(image, central_fraction=0.875):
    """Load and preprocess an image.

    Args:
      image: a tf.string tensor with an JPEG-encoded image.
      central_fraction: do a central crop with the specified
        fraction of image covered.
    Returns:
      An ops.Tensor that produces the preprocessed image.
    """

    # Decode Jpeg data and convert to float.
    image = tf.cast(tf.image.decode_jpeg(image, channels=3), tf.float32)

    image = tf.image.central_crop(image, central_fraction=central_fraction)
    # Make into a 4D tensor by setting a 'batch size' of 1.
    image = tf.expand_dims(image, [0])
    image = tf.image.resize_bilinear(image,
                                     [FLAGS.image_size, FLAGS.image_size],
                                     align_corners=False)

    # Center the image about 128.0 (which is done during training) and normalize.
    image = tf.multiply(image, 1.0 / 127.5)
    return tf.subtract(image, 1.0)


def main(args):
    if not os.path.exists(FLAGS.checkpoint):
        tf.logging.fatal(
            'Checkpoint %s does not exist. Have you download it? See tools/download_data.sh',
            FLAGS.checkpoint)
    g = tf.Graph()
    with g.as_default():
        input_image = tf.placeholder(tf.string)
        processed_image = PreprocessImage(input_image)

        with slim.arg_scope(inception.inception_v3_arg_scope()):
            logits, end_points = inception.inception_v3(
                processed_image, num_classes=FLAGS.num_classes, is_training=False)

        predictions = end_points['multi_predictions'] = tf.nn.sigmoid(
            logits, name='multi_predictions')

        sess = tf.Session()

        saver = tf_saver.Saver()

        logits_2 = layers.conv2d(
            end_points['PreLogits'],
            FLAGS.num_classes, [1, 1],
            activation_fn=None,
            normalizer_fn=None,
            scope='Conv2d_final_1x1')

        logits_2 = array_ops.squeeze(logits_2, [1, 2], name='SpatialSqueeze_2')

        predictions_2 = end_points['multi_predictions_2'] = tf.nn.sigmoid(logits_2, name='multi_predictions_2')

        sess.run(tf.global_variables_initializer())

        saver.restore(sess, FLAGS.checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='2016_08/model.ckpt',
                        help='Checkpoint to run inference on.')
    parser.add_argument('--dict', type=str, default='dict.csv',
                        help='Path to a dict.csv that translates from mid to a display name.')
    parser.add_argument('--image_size', type=int, default=299,
                        help='Image size to run inference on.')
    parser.add_argument('--num_classes', type=int, default=6012,
                        help='Number of output classes.')
    parser.add_argument('--image_path', default='test_set/0a9ed4def08fe6d1')
    FLAGS = parser.parse_args()
    tf.app.run()
