import tensorflow as tf
import constants
from base_fns import FunctionDispatcher


class CopyFromOutput(FunctionDispatcher):
    """
    Copy a particular output from the given layer
    - layer might be different in the eval mode
    - (testing) with some kind of normalization applied
    """
    def __init__(self, *args, **kwargs):
        super(CopyFromOutput, self).__init__(*args, **kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def make_call(self, features, train_attention_to_copy, eval_attention_to_copy):
        if self.teacher_forcing:
            attention_to_copy = tf.one_hot(
                train_attention_to_copy, tf.shape(train_attention_to_copy)[-1],
                on_value=constants.VERY_LARGE,
                off_value=constants.VERY_SMALL
            )
        else:
            attention_to_copy = eval_attention_to_copy

        # return tf.nn.softmax(attention_to_copy)
        return attention_to_copy


class LabelAttention(FunctionDispatcher):
    def __init__(self, *args, **kwargs):
        super(LabelAttention, self).__init__(*args, **kwargs)

    def make_call(self, features, train_label_scores, eval_label_scores, label_embeddings):
        embeddings_shape = label_embeddings.get_shape()
        vocab_size = embeddings_shape[0]
        label_embedding_dim = embeddings_shape[1]
        input_shape = tf.shape(train_label_scores)
        batch_size = input_shape[0]
        batch_seq_len = input_shape[1]

        label_scores = train_label_scores if self.teacher_forcing else eval_label_scores

        # check whether this thing is actually scores or if it's predictions, and needs
        # to be expanded out to one-hot scores. If it's actually scores, dims should be
        # batch x batch_seq_len x num_classes, and thus rank should be 3
        if len(label_scores.get_shape()) < 3:
            label_scores = tf.one_hot(label_scores, vocab_size)

        label_scores = tf.reshape(label_scores, [-1, vocab_size])
        label_embeddings = tf.reshape(label_embeddings, [vocab_size, label_embedding_dim])
        averaged = tf.matmul(label_scores, label_embeddings)

        return tf.reshape(averaged, [batch_size, batch_seq_len, label_embedding_dim])


dispatcher = {
    'copy_from_predicted': CopyFromOutput,
    'label_attention': LabelAttention,
}
