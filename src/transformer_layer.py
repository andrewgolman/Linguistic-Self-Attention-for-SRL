import tensorflow as tf
import tensorflow.keras as keras
import evaluation_fns
import attention_fns
from opennmt.layers import transformer as onmt_transformer
from opennmt.layers.transformer import SelfAttentionEncoderLayer, MultiHeadAttention
# https://github.com/OpenNMT/OpenNMT-tf/blob/master/opennmt/layers/transformer.py#L339


class MultiHeadAttentionWithSpecial(MultiHeadAttention):

    def call(self, inputs, specials=(), memory=None, mask=None, cache=None, training=None):
        """
        Runs the layer.
        This code is taken from the base class, with alterations regarding only special values
        """
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

        attn = tf.cast(tf.nn.softmax(tf.cast(dot, tf.float32)), dot.dtype)

        special_attn, special_values = specials
        # <AG CODE HERE>
        attn = tf.concat([attn] + list(map(lambda x: tf.expand_dims(x, 1), special_attn)), axis=1)
        values = tf.concat(special_values + [values], axis=1)

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
    def __init__(self, num_heads, num_units, attention_dropout, maximum_relative_position, **kwargs):
        super(SelfAttentionEncoderLayer, self).__init__(**kwargs)
        self.self_attention = MultiHeadAttentionWithSpecial(
            num_heads,
            num_units,
            dropout=attention_dropout,
            maximum_relative_position=maximum_relative_position
         )


class TransformerLayer(keras.models.Model):
    def __init__(self, transformer_layer_id, task_config, layer_config, hparams):
        super(TransformerLayer, self).__init__()
        self.transformer_layer_id = transformer_layer_id
        self.task_config = task_config
        self.layer_config = layer_config
        self.hparams = hparams

        self.attention_fns = []
        self.value_fns = []

        self.sa_layer = SelfAttentionEncoderLayerWithSpecial(  # todo AG 2-layer FFN vs 3 in original LISA
            num_heads=self.layer_config['num_heads'],
            num_units=self.layer_config['head_dim'],
            ffn_inner_dim=self.layer_config['ff_hidden_size'],
            dropout=self.hparams.prepost_dropout,  # todo AG check after add vs before add
            attention_dropout=self.hparams.attn_dropout,
            ffn_dropout=self.hparams.ff_dropout,
            ffn_activation=tf.nn.leaky_relu,  # todo AG do we need it to be leaky?
            maximum_relative_position=None,
        )

        if self.transformer_layer_id in self.attention_config:
            this_layer_attn_config = self.attention_config[self.transformer_layer_id]

            for attn_fn, attn_fn_map in this_layer_attn_config.get(['attention_fns'], {}).items():
                self.attention_fns.append(
                    attention_fns.dispatcher(attn_fn_map['name'])(attn_fn_map)
                )

            for value_fn, value_fn_map in this_layer_attn_config.get(['value_fns'], {}).items():
                self.value_fns.append(
                    attention_fns.dispatcher(value_fn_map['name'])(value_fn_map)
                )

    def apply_special_attention(self, features, mask, outputs):
        special_attn = []
        special_values = []

        for attn_fn in self.attention_fns:
            fn_values = attn_fn(features, outputs=outputs)
            special_attn.append(fn_values)

        for value_fn in self.value_fns:
            fn_values = value_fn(features, outputs=outputs)
            special_values.append(fn_values)

        return special_attn, special_values

    def call(self, data):
        features, mask, outputs = data
        special_attn, special_values = self.special_attention(features=features, outputs=outputs)

        features = self.sa_layer(inputs=features, mask=mask, specials=(special_attn, special_values))
        return features
