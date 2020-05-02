import tensorflow as tf
import tensorflow.keras as keras
from opennmt.layers import transformer as onmt_transformer
from opennmt.layers.transformer import SelfAttentionEncoderLayer, MultiHeadAttention, TransformerLayerWrapper
# https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/layers/transformer.py#L339


class MultiHeadAttentionWithSpecial(MultiHeadAttention):

    def call(self, inputs, memory=None, mask=None, cache=None, training=None, specials=None):
        """
        Runs the layer.
        This code is taken from the base class, with alterations regarding only special values
        Special attention heads and values replace normally calculated heads, so model can utilize outside data
        """

        special_attn, special_values = specials

        def _compute_kv(x):
            keys = self.linear_keys(x)
            keys = onmt_transformer.split_heads(keys, self.num_heads)
            values = self.linear_values(x)
            values = onmt_transformer.split_heads(values, self.num_heads)
            return keys, values

        # Compute queries.
        queries = self.linear_queries(inputs)
        queries = onmt_transformer.split_heads(queries, self.num_heads)
        queries *= self.num_units_per_head**-0.5

        # Compute keys and values.
        if memory is None:
            keys, values = _compute_kv(inputs)
            if cache:
                keys = tf.concat([cache[0], keys], axis=2)
                values = tf.concat([cache[1], values], axis=2)
        else:
            if cache:
                keys, values = tf.cond(
                    tf.equal(tf.shape(cache[0])[2], 0),
                    true_fn=lambda: _compute_kv(memory),
                    false_fn=lambda: cache)
            else:
                keys, values = _compute_kv(memory)

        if self.maximum_relative_position is not None:
            if memory is not None:
                raise ValueError("Relative position representations only supports self-attention")
            keys_length = tf.shape(keys)[2]
            relative_pos = onmt_transformer.relative_positions(
                keys_length,
                self.maximum_relative_position,
                with_cache=bool(cache))
            relative_repr_keys = tf.gather(self.relative_position_keys, relative_pos)
            relative_repr_values = tf.gather(self.relative_position_values, relative_pos)
        else:
            relative_repr_keys = None
            relative_repr_values = None

        cache = (keys, values)

        # Dot product attention.
        dot = tf.matmul(queries, keys, transpose_b=True)
        if relative_repr_keys is not None:
            dot += onmt_transformer.matmul_with_relative_representations(queries, relative_repr_keys, transpose_b=True)
        if mask is not None:
            mask = tf.cast(mask, tf.float32)
            if mask.shape.rank == 2:
                mask = tf.expand_dims(mask, 1)  # Broadcast on time dimension.
            mask = tf.expand_dims(mask, 1)  # Broadcast on head dimension.
            dot = tf.cast(tf.cast(dot, tf.float32) * mask + ((1.0 - mask) * tf.float32.min), dot.dtype)


        # Replace last heads with special heads. todo AG optimize
        if len(special_attn) > 0:
            unstacked_dot = tf.unstack(dot, axis=1)  # [BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN]
            for i in range(len(special_attn)):q
                unstacked_dot[-i] = tf.zeros_like(unstacked_dot[-i])
            dot = tf.stack(unstacked_dot, axis=1)
            attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)
            unstacked_attn = tf.unstack(attn, axis=1)  # [BATCH_SIZE, HEADS, SEQ_LEN, SEQ_LEN]
            for i, t in enumerate(special_attn):
                unstacked_attn[-i] = t
            attn = tf.stack(unstacked_attn, axis=1)

        else:
            attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)

        if len(special_values) > 0:
            unstacked_values = tf.unstack(values, axis=1)
            for i, t in enumerate(special_values):
                unstacked_values[-i] = t
            values = tf.stack(unstacked_values, axis=1)

        drop_attn = onmt_transformer.common.dropout(attn, self.dropout, training=training)
        heads = tf.matmul(drop_attn, values)
        if relative_repr_values is not None:
            heads += onmt_transformer.matmul_with_relative_representations(drop_attn, relative_repr_values)

        # Concatenate all heads output.
        combined = onmt_transformer.combine_heads(heads)
        outputs = self.linear_output(combined)
        if self.return_attention:
            return outputs, cache, attn
        return outputs, cache


class SelfAttentionEncoderLayerWithSpecial(SelfAttentionEncoderLayer):
    """
    Replace self.self_attention with the attention, derived from MultiHeadAttention
    """
    def __init__(self,
                 num_units,
                 num_heads,
                 ffn_inner_dim,
                 dropout=0.1,
                 attention_dropout=0.1,
                 ffn_dropout=0.1,
                 ffn_activation=tf.nn.relu,
                 **kwargs):
        super(SelfAttentionEncoderLayerWithSpecial, self).__init__(
            num_units, num_heads, ffn_inner_dim,
            dropout, attention_dropout,
            ffn_dropout, ffn_activation,
            **kwargs
        )
        self.self_attention = MultiHeadAttentionWithSpecial(
            num_heads,
            num_units,
            dropout=attention_dropout,
         )
        self.self_attention = TransformerLayerWrapper(
            self.self_attention, dropout
        )

    def call(self, x, mask=None, training=None, specials=None):  # pylint: disable=arguments-differ
        """Runs the encoder layer."""
        y, _ = self.self_attention(x, mask=mask, training=training, specials=specials)
        y = self.ffn(y, training=training)
        return y


class TransformerLayer(keras.layers.Layer):
    def __init__(self, transformer_layer_id, layer_config, attn_fns, val_fns, hparams):
        super(TransformerLayer, self).__init__()
        self.transformer_layer_id = transformer_layer_id

        self.layer_config = layer_config
        self.hparams = hparams

        self.attention_fns = attn_fns
        self.value_fns = val_fns

        self.sa_layer = SelfAttentionEncoderLayerWithSpecial(  # todo AG 2-layer FFN vs 3 in original LISA
            num_heads=self.layer_config['num_heads'],
            num_units=self.layer_config['head_dim'] * self.layer_config['num_heads'],
            ffn_inner_dim=self.layer_config['ff_hidden_size'],
            dropout=1-self.hparams.prepost_dropout,
            attention_dropout=1-self.hparams.attn_dropout,
            ffn_dropout=1-self.hparams.ff_dropout,
            ffn_activation=tf.nn.leaky_relu,  # todo verify do we need it to be leaky?
        )

    def compute_special_attention(self, features, mask, outputs, labels):
        """
        :return: todo doc
         special_attn: List[[]]
         special_values: List[[]]
        """
        special_attn = []
        special_values = []

        for attn_fn in self.attention_fns:
            fn_values = attn_fn(features, outputs=outputs, labels=labels)
            special_attn.append(fn_values)

        for value_fn in self.value_fns:
            fn_values = value_fn(features, outputs=outputs, labels=labels)
            special_values.append(fn_values)

        return special_attn, special_values

    def call(self, data):
        """
        features: [BATCH_SIZE, SEQ_LEN, SA_HID]
        mask: [BATCH_SIZE, SEQ_LEN]
        outputs: Dict{task: task_outputs}
        labels: Dict: {task : [BATCH_SIZE, SEQ_LEN]} (for srl: [..., 9])
        :return features: [BATCH_SIZE, SEQ_LEN, SA_HID]
        """
        features, mask, outputs, labels = data
        special_attn, special_values = self.compute_special_attention(features, mask, outputs, labels)

        features = self.sa_layer(features, mask=mask, specials=(special_attn, special_values))
        return features

    def enable_teacher_forcing(self):
        for f in self.attention_fns:
            f.enable_teacher_forcing()
        for f in self.value_fns:
            f.enable_teacher_forcing()

    def disable_teacher_forcing(self):
        for f in self.attention_fns:
            f.disable_teacher_forcing()
        for f in self.value_fns:
            f.disable_teacher_forcing()
