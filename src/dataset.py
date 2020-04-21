# tf check line 79f

import tensorflow as tf
import constants
from data_generator import conll_data_generator


def map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names):
  def _mapper(d):
    intmapped = []
    for i, datum_name in enumerate(feature_label_names):
      if 'vocab' in data_config[datum_name]:
        # todo this is a little clumsy -- is there a better way to pass this info through?
        # todo also we need the variable-length feat to come last, gross
        if 'type' in data_config[datum_name] and data_config[datum_name]['type'] == 'range':
          idx = data_config[datum_name]['conll_idx']
          if idx[1] == -1:
            intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:]))
          else:
            last_idx = i + idx[1]
            intmapped.append(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i:last_idx]))
        else:
          intmapped.append(tf.expand_dims(vocab_lookup_ops[data_config[datum_name]['vocab']].lookup(d[:, i]), -1))
      else:
        intmapped.append(tf.expand_dims(tf.strings.to_number(d[:, i], out_type=tf.int64), -1))

    # this is where the order of features/labels in input gets defined
    # todo: can i have these come out of the lookup as int32?
    return tf.cast(tf.concat(intmapped, axis=-1), tf.int32)

  return _mapper


def create_dataset(data_filenames, data_config, vocab_lookup_ops):
    # get the names of data fields in data_config that correspond to features or labels,
    # and thus that we want to load into batches
    feature_label_names = [d for d in data_config.keys() if \
                           ('feature' in data_config[d] and data_config[d]['feature']) or
                           ('label' in data_config[d] and data_config[d]['label'])]

    # get the dataset
    dataset = tf.data.Dataset.from_generator(lambda: conll_data_generator(data_filenames, data_config),
                                             output_shapes=[None, None], output_types=tf.string)

    # intmap the dataset
    dataset = dataset.map(map_strings_to_ints(vocab_lookup_ops, data_config, feature_label_names), num_parallel_calls=8)
    return dataset


def get_dataset(data_filenames, data_config, vocab_lookup_ops, batch_size, num_epochs, shuffle,
                shuffle_buffer_multiplier=None):

  bucket_boundaries = constants.DEFAULT_BUCKET_BOUNDARIES
  bucket_batch_sizes = [batch_size] * (len(bucket_boundaries) + 1)

  # todo do something smarter with multiple files + parallel?

  with tf.device('/cpu:0'):
    dataset = create_dataset(data_filenames, data_config, vocab_lookup_ops)
    # dataset = dataset.cache()  # todo AG

    # do batching
    dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(element_length_func=lambda d: tf.shape(d)[0],
                                                                      bucket_boundaries=bucket_boundaries,
                                                                      bucket_batch_sizes=bucket_batch_sizes,
                                                                      padded_shapes=tf.compat.v1.data.get_output_shapes(dataset),
                                                                      padding_values=constants.PAD_VALUE))

    # shuffle and expand out epochs if training
    if shuffle:
      dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=batch_size*shuffle_buffer_multiplier,
                                                                 count=num_epochs))

    # todo should the buffer be bigger?
    dataset.prefetch(buffer_size=1)

    return dataset
