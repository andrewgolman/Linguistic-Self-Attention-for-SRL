import tensorflow as tf
import numpy as np
import tensorflow.keras.layers as L
from opennmt.utils.misc import shape_list


def int_to_str_lookup_table(inputs, lookup_map):
    # todo order of map.values() is probably not guaranteed; should prob sort by keys first
    return tf.nn.embedding_lookup(np.array(list(lookup_map.values())), inputs)


# similar to https://github.com/keras-team/keras/blob/master/keras/layers/core.py#L796
class Bilinear(L.Layer):
    def __init__(self, output_size, add_bias=True):
        super(Bilinear, self).__init__()
        self.output_size = output_size
        self.add_bias = add_bias

    def build(self, input_shape):
        self.matrix_shape = input_shape[0][-1] + self.add_bias
        self.left_dense = L.Dense(self.output_size * self.matrix_shape)
        super(Bilinear, self).build(input_shape)

    def call(self, data):
        left, right = data

        if self.add_bias:
            left_bias_shape = shape_list(left)
            left_bias_shape[-1] = 1
            right_bias_shape = shape_list(right)
            right_bias_shape[-1] = 1
            left = tf.concat([left, tf.ones(left_bias_shape, dtype=tf.float32)], axis=2)
            right = tf.concat([right, tf.ones(right_bias_shape, dtype=tf.float32)], axis=2)

        lin = self.left_dense(left)
        lin_shape = shape_list(lin)
        lin = tf.reshape(lin, [-1, lin_shape[1] * self.output_size, lin_shape[2] // self.output_size])
        bilin = tf.matmul(lin, right, transpose_b=True)
        bilin_shape = shape_list(bilin)
        bilin = tf.reshape(bilin, [bilin_shape[0], bilin_shape[1] // self.output_size, self.output_size, bilin_shape[-1]])
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
        super(ConditionalBilinearClassifier, self).__init__()
        self.left_dropout = L.Dropout(dropout, noise_shape=[None, 1, left_input_size])
        self.right_dropout = L.Dropout(dropout, noise_shape=[None, 1, right_input_size])
        self.bilinear = Bilinear(n_outputs)
        self.n_outputs = n_outputs

    def call(self, data):
        left, right, probs = data
        # left: [BATCH_SIZE, SEQ_LEN, HID]
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
