import tensorflow as tf


class AttentionMechanism(object):


    def __init__(self, img, dim_e, tiles=1):

        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N    = tf.shape(input=img)[0]
            H, W = tf.shape(input=img)[1], tf.shape(input=img)[2] # image
            C    = img.shape[3]                # channels
            self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError

        # dimensions
        self._n_regions  = tf.shape(input=self._img)[1]
        self._n_channels = self._img.shape[2]
        self._dim_e      = dim_e
        self._tiles      = tiles
        self._scope_name = "att_mechanism"

        # attention vector over the image
        self._att_img = tf.compat.v1.layers.dense(
            inputs=self._img,
            units=self._dim_e,
            use_bias=False,
            name="att_img")


    def context(self, h):

        with tf.compat.v1.variable_scope(self._scope_name):
            if self._tiles > 1:
                att_img = tf.expand_dims(self._att_img, axis=1)
                att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
                att_img = tf.reshape(att_img, shape=[-1, self._n_regions,
                        self._dim_e])
                img = tf.expand_dims(self._img, axis=1)
                img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
                img = tf.reshape(img, shape=[-1, self._n_regions,
                        self._n_channels])
            else:
                att_img = self._att_img
                img     = self._img

            # computes attention over the hidden vector
            att_h = tf.compat.v1.layers.dense(inputs=h, units=self._dim_e, use_bias=False)

            # sums the two contributions
            att_h = tf.expand_dims(att_h, axis=1)
            att = tf.tanh(att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.compat.v1.get_variable("att_beta", shape=[self._dim_e, 1],
                    dtype=tf.float32)
            att_flat = tf.reshape(att, shape=[-1, self._dim_e])
            e = tf.matmul(att_flat, att_beta)
            e = tf.reshape(e, shape=[-1, self._n_regions])

            # compute weights
            a = tf.nn.softmax(e)
            a = tf.expand_dims(a, axis=-1)
            c = tf.reduce_sum(input_tensor=a * img, axis=1)

            return c


    def initial_cell_state(self, cell):

        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell


    def initial_state(self, name, dim):

        with tf.compat.v1.variable_scope(self._scope_name):
            img_mean = tf.reduce_mean(input_tensor=self._img, axis=1)
            W = tf.compat.v1.get_variable("W_{}_0".format(name), shape=[self._n_channels,
                    dim])
            b = tf.compat.v1.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h