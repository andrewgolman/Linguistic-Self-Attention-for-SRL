import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as K
import tensorflow.keras.layers as L
from opennmt.utils.misc import shape_list


def int_to_str_lookup_table(inputs, lookup_map):
  # todo order of map.values() is probably not guaranteed; should prob sort by keys first
  return tf.nn.embedding_lookup(np.array(list(lookup_map.values())), inputs)


def set_vars_to_moving_average(moving_averager):
  moving_avg_variables = tf.get_collection(tf.GraphKeys.MOVING_AVERAGE_VARIABLES)
  return tf.group(*[tf.assign(x, moving_averager.average(x)) for x in moving_avg_variables])


# similar to https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L796
class Bilinear(L.Layer):
    def __init__(self, output_size, add_bias=True):
        # output_size = 2
        super(Bilinear, self).__init__()
        self.output_size = output_size
        self.add_bias = add_bias

    def build(self, input_shape):
        # todo AG check
        self.matrix_shape = input_shape[0][-1] + self.add_bias
        self.kernel = self.add_weight(
            name='kernel',
            # inputs1_size + add_bias1, output_size, inputs2_size + add_bias2
            shape=(self.matrix_shape, self.output_size, self.matrix_shape),
            initializer='glorot_uniform',
            trainable=True
        )
        super(Bilinear, self).build(input_shape)

    def call(self, data):
        left, right = data

        if self.add_bias:
            left_bias_shape = shape_list(left)
            left_bias_shape[-1] = 1
            right_bias_shape = shape_list(right)
            right_bias_shape[-1] = 1
            left = K.concatenate([left, tf.ones(left_bias_shape, dtype=tf.float32)], axis=-1)
            right = K.concatenate([right, tf.ones(right_bias_shape, dtype=tf.float32)], axis=-1)

        lin = tf.matmul(left, tf.reshape(self.kernel, [self.matrix_shape, -1]))
        lin_shape = shape_list(lin)
        lin = K.reshape(lin, [-1, lin_shape[1] * self.output_size, lin_shape[2] // self.output_size])
        right = tf.transpose(right, [0, 2, 1])
        bilin = tf.matmul(lin, right)
        bilin_shape = shape_list(bilin)
        bilin = K.reshape(bilin, [bilin_shape[0], bilin_shape[1] // self.output_size, self.output_size, bilin_shape[-1]])
        return bilin


class BilinearClassifier(L.Layer):
    def __init__(self, n_outputs, dropout=0, left_input_size=1, right_input_size=1):
        super(BilinearClassifier, self).__init__()
        self.bilinear = Bilinear(n_outputs)
        self.left_dropout = L.Dropout(dropout, noise_shape=[None, 1, left_input_size])
        self.right_dropout = L.Dropout(dropout, noise_shape=[None, 1, right_input_size])

    def call(self, data):
        left, right = data
        left = self.left_dropout(left)
        right = self.right_dropout(right)
        bilin = self.bilinear([left, right])
        return bilin


class ConditionalBilinearClassifier(L.Layer):
    def __init__(self, n_outputs, dropout, left_input_size, right_input_size):
        #     noise_shape = tf.stack([batch_size, 1, input_size])
        super(ConditionalBilinearClassifier, self).__init__()
        self.left_dropout = L.Dropout(dropout, noise_shape=[None, 1, left_input_size])
        self.right_dropout = L.Dropout(dropout, noise_shape=[None, 1, right_input_size])
        self.bilinear = Bilinear(n_outputs)
        self.n_outputs = n_outputs

    def call(self, data):
        left, right, probs = data
        # left: [BATCH_SIZE, SEQ_LEN, HID]  todo AG check all calls
        # right: [BATCH_SIZE, SEQ_LEN, HID]
        # probs: [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
        left = self.left_dropout(left)
        right = self.right_dropout(right)
        bilin = self.bilinear([left, right])

        input_shape = tf.shape(left)
        batch_size = input_shape[0]
        bucket_size = input_shape[1]

        if len(probs.get_shape().as_list()) == 2:
            probs = tf.cast(tf.one_hot(tf.cast(probs, dtype=tf.int64), bucket_size, 1, 0), dtype=tf.float32)
        else:
            probs = tf.stop_gradient(probs)

        bilin = tf.reshape(bilin, [batch_size, bucket_size, self.n_outputs, bucket_size])
        weighted_bilin = tf.squeeze(tf.matmul(bilin, tf.expand_dims(probs, 3)), -1)  # [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
        return weighted_bilin, bilin
