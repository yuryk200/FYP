import numpy as np
import tensorflow as tf


from .components.positional import add_timing_signal_nd


class Encoder(object):
    """Class with a __call__ method that applies convolutions to an image"""

    def __init__(self, config):
        self._config = config


    def __call__(self, training, img, dropout):
        """Applies convolutions to the image

        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels), of type
                tf.uint8

        Returns:
            the encoded images, shape = (?, h', w', c')

        """
        img = tf.cast(img, tf.float32) / 255.

        with tf.compat.v1.variable_scope("convolutional_encoder"):
            # conv + max pool -> /2
            out = tf.compat.v1.layers.conv2d(img, 64, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.compat.v1.layers.max_pooling2d(out, 2, 2, "SAME")

            # conv + max pool -> /2
            out = tf.compat.v1.layers.conv2d(out, 128, 3, 1, "SAME",
                    activation=tf.nn.relu)
            out = tf.compat.v1.layers.max_pooling2d(out, 2, 2, "SAME")

            # regular conv -> id
            out = tf.compat.v1.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)

            out = tf.compat.v1.layers.conv2d(out, 256, 3, 1, "SAME",
                    activation=tf.nn.relu)

            if self._config.encoder_cnn == "vanilla":
                out = tf.compat.v1.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

            out = tf.compat.v1.layers.conv2d(out, 512, 3, 1, "SAME",
                    activation=tf.nn.relu)

            if self._config.encoder_cnn == "vanilla":
                out = tf.compat.v1.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

            if self._config.encoder_cnn == "cnn":
                # conv with stride /2 (replaces the 2 max pool)
                out = tf.compat.v1.layers.conv2d(out, 512, (2, 4), 2, "SAME")

            # conv
            out = tf.compat.v1.layers.conv2d(out, 512, 3, 1, "VALID",
                    activation=tf.nn.relu)

            if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
                out = add_timing_signal_nd(out)

        return out