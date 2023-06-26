import tensorflow as tf
import collections

from tensorflow.compat.v1.nn.rnn_cell import RNNCell, LSTMStateTuple


AttentionState = collections.namedtuple("AttentionState", ("cell_state", "o"))


class AttentionCell(RNNCell):
    def __init__(self, cell, attention_mechanism, dropout, attn_cell_config,
        num_proj, dtype=tf.float32):

        # variables and tensors
        self._cell                = cell
        self._attention_mechanism = attention_mechanism
        self._dropout             = dropout

        # hyperparameters and shapes
        self._n_channels     = self._attention_mechanism._n_channels
        self._dim_e          = attn_cell_config["dim_e"]
        self._dim_o          = attn_cell_config["dim_o"]
        self._num_units      = attn_cell_config["num_units"]
        self._dim_embeddings = attn_cell_config["dim_embeddings"]
        self._num_proj       = num_proj
        self._dtype          = dtype

        # for RNNCell
        self._state_size = AttentionState(self._cell._state_size, self._dim_o)


    @property
    def state_size(self):
        return self._state_size


    @property
    def output_size(self):
        return self._num_proj


    @property
    def output_dtype(self):
        return self._dtype


    def initial_state(self):

        initial_cell_state = self._attention_mechanism.initial_cell_state(self._cell)
        initial_o          = self._attention_mechanism.initial_state("o", self._dim_o)

        return AttentionState(initial_cell_state, initial_o)


    def step(self, embedding, attn_cell_state):

        prev_cell_state, o = attn_cell_state

        scope = tf.compat.v1.get_variable_scope()
        with tf.compat.v1.variable_scope(scope):
            # compute new h
            x                     = tf.concat([embedding, o], axis=-1)
            new_h, new_cell_state = self._cell.__call__(x, prev_cell_state)
            new_h = tf.nn.dropout(new_h, rate=1 - (self._dropout))

            # compute attention
            c = self._attention_mechanism.context(new_h)

            # compute o
            o_W_c = tf.compat.v1.get_variable("o_W_c", dtype=tf.float32,
                    shape=(self._n_channels, self._dim_o))
            o_W_h = tf.compat.v1.get_variable("o_W_h", dtype=tf.float32,
                    shape=(self._num_units, self._dim_o))

            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(c, o_W_c))
            new_o = tf.nn.dropout(new_o, rate=1 - (self._dropout))

            y_W_o = tf.compat.v1.get_variable("y_W_o", dtype=tf.float32,
                    shape=(self._dim_o, self._num_proj))
            logits = tf.matmul(new_o, y_W_o)

            # new Attn cell state
            new_state = AttentionState(new_cell_state, new_o)

            return logits, new_state


    def __call__(self, inputs, state):

        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)