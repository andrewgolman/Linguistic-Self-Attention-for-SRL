import tensorflow as tf
import constants
from data_generator import conll_data_generator


def get_field_mapper(vocab_lookup_ops, data_config, feature_label_names):
    def _mapper(row):
        mapped = []
        for i, datum_name in enumerate(feature_label_names):
            if 'vocab' in data_config[datum_name]:
                # this is a little clumsy -- is there a better way to pass this info through?
                # also we need the variable-length feat to come last, gross
                if data_config[datum_name].get('type') == 'range':
                    idx = data_config[datum_name]['conll_idx']
                    if idx[1] == -1:
                        mapped.append(tf.cast(
                            vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(row[:, i:]),
                        tf.float32))
                    else:
                        last_idx = i + idx[1]
                        mapped.append(tf.cast(
                            vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(row[:, i:last_idx]),
                        tf.float32))
                else:
                    mapped.append(tf.cast(
                        tf.expand_dims(
                            vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(row[:, i]), -1
                        ),
                        tf.float32))
            # elif 'tokenizer' in data_config[datum_name]:
            #     tokenizer = preprocessor_maps.tokenizers[data_config[datum_name]['tokenizer']]
            #     tokens = tf.map_fn(lambda s: tokenize(tokenizer, s), row[:, i], tf.int64)
            #     mapped.append(
            #         tf.expand_dims(
            #             tf.convert_to_tensor(tokens, dtype=tf.int64), -1
            #         )
            #     )
            else:
                mapped.append(tf.expand_dims(tf.strings.to_number(row[:, i], out_type=tf.float32), -1))

        # this is where the order of features/labels in input gets defined
        return tf.cast(tf.concat(mapped, axis=-1), tf.float32)

    return _mapper


def create_dataset(data_filenames, data_config, vocab_lookup_ops):
    # get the names of data fields in data_config that correspond to features or labels,
    # and thus that we want to load into batches
    feature_label_names = [d for d in data_config.keys() if data_config[d].get('feature') or data_config[d].get('label')]

    # get the dataset
    dataset = tf.data.Dataset.from_generator(lambda: conll_data_generator(data_filenames, data_config),
                                             output_shapes=[None, None], output_types=tf.string)

    # intmap the dataset
    mapper = get_field_mapper(vocab_lookup_ops, data_config, feature_label_names)
    dataset = dataset.map(mapper, num_parallel_calls=8)
    return dataset


def get_dataset(data_filenames, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
                shuffle_buffer_multiplier=None):

  bucket_boundaries = constants.DEFAULT_BUCKET_BOUNDARIES
  bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

  with tf.device('/cpu:0'):
    dataset = create_dataset(data_filenames, data_config, vocab_lookup_ops)
    dataset = dataset.cache()

    # do batching
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
        element_length_func=lambda d: tf.shape(d)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        padded_shapes=tf.compat.v1.data.get_output_shapes(dataset),
        padding_values=float(constants.PAD_VALUE)
    ))

    # shuffle and expand out epochs if training
    if shuffle:
      dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size*shuffle_buffer_multiplier,
                                                                 count=num_epochs))

    dataset.prefetch(buffer_size=256)

    return dataset
