import numpy as np
import tensorflow as tf
import sys
import tensorflow.compat.v1.logging as logging


def fatal_error(message):
    tf.compat.v1.logging.error(message)
    sys.exit(1)


def init_logging(verbosity):
    logging.set_verbosity(verbosity)
    logging.log(logging.INFO, "Using Python version %s" % sys.version)
    logging.log(logging.INFO, "Using TensorFlow version %s" % tf.__version__)


def batch_str_decode(string_array, codec='utf-8'):
  string_array = np.array(string_array)
  return np.reshape(np.array(list(map(lambda p: p if not p or isinstance(p, str) else p.decode(codec),
                             np.reshape(string_array, [-1])))), string_array.shape)

def load_transitions(transition_statistics, num_classes, vocab_map):
  transition_statistics_np = np.zeros((num_classes, num_classes))
  with open(transition_statistics, 'r') as f:
    for line in f:
      tag1, tag2, prob = line.split("\t")
      transition_statistics_np[vocab_map[tag1], vocab_map[tag2]] = float(prob)
    logging.log(logging.INFO, "Loaded pre-computed transition statistics: %s" % transition_statistics)
  return transition_statistics_np


def load_pretrained_embeddings(pretrained_fname: str) -> np.array:
  """
  Load float matrix from one file
  """
  logging.log(logging.INFO, "Loading pre-trained embedding file: %s" % pretrained_fname)

  # TODO: np.loadtxt refuses to work for some reason
  # pretrained_embeddings = np.loadtxt(self.args.word_embedding_file, usecols=range(1, word_embedding_size+1))

  pretrained_embeddings = []
  with open(pretrained_fname, 'r') as f:
    for line in f:
      embedding = [float(s) for s in line.split()[1:]]
      pretrained_embeddings.append(embedding)

  pretrained_embeddings = np.array(pretrained_embeddings)
  pretrained_embeddings /= np.std(pretrained_embeddings)
  return pretrained_embeddings


def load_transition_params(task_config, vocab):
  transition_params = {}
  for layer, task_maps in task_config.items():
    for task, task_map in task_maps.items():
      task_crf = 'crf' in task_map and task_map['crf']
      task_viterbi_decode = task_crf or 'viterbi' in task_map and task_map['viterbi']
      if task_viterbi_decode:
        transition_params_file = task_map['transition_stats'] if 'transition_stats' in task_map else None
        if not transition_params_file:
          fatal_error("Failed to load transition stats for task '%s' with crf=%r and viterbi=%r" %
                      (task, task_crf, task_viterbi_decode))
        if transition_params_file and task_viterbi_decode:
          transitions = load_transitions(transition_params_file, vocab.vocab_names_sizes[task],
                                              vocab.vocab_maps[task])
          transition_params[task] = transitions
  return transition_params


def load_feat_label_idx_maps(data_config):
  feature_idx_map = {}
  label_idx_map = {}
  for i, f in enumerate([d for d in data_config.keys() if
                       ('feature' in data_config[d] and data_config[d]['feature']) or
                       ('label' in data_config[d] and data_config[d]['label'])]):
    if 'feature' in data_config[f] and data_config[f]['feature']:
      feature_idx_map[f] = i
    if 'label' in data_config[f] and data_config[f]['label']:
      if 'type' in data_config[f] and data_config[f]['type'] == 'range':
        idx = data_config[f]['conll_idx']
        j = i + idx[1] if idx[1] != -1 else -1
        label_idx_map[f] = (i, j)
      else:
        label_idx_map[f] = (i, i+1)
  return feature_idx_map, label_idx_map


def combine_attn_maps(layer_config, attention_config, task_config):
  layer_task_config = {}
  layer_attention_config = {}
  for task_or_attn_name, layer in layer_config.items():
    if task_or_attn_name in attention_config:
      layer_attention_config[layer] = attention_config[task_or_attn_name]
    elif task_or_attn_name in task_config:
      if layer not in layer_task_config:
        layer_task_config[layer] = {}
      layer_task_config[layer][task_or_attn_name] = task_config[task_or_attn_name]
    else:
      fatal_error('No task or attention config "%s"' % task_or_attn_name)
  return layer_task_config, layer_attention_config


def list2dict(l, keys):
    # assert len(l) == len(keys), (len(l), keys)
    keys = sorted(keys)
    return {keys[i]: l[i] for i in range(len(keys))}


def task_list(task_config):
    return sum([list(v.keys()) for v in task_config.values()], [])


# import tensorflow.python.training.tracking.tracking as tracking

# https://github.com/tensorflow/tensorflow/blob/c3973c78f03c50d8514c14c2866ab30e708aea24/tensorflow/python/training/tracking/tracking.py
# class NotTrackableDict(tracking.NotTrackable, dict):
class NotTrackableDict(dict):
    def __init__(self, data):
        # tracking.NotTrackable.__init__(self)
        dict.__init__(self, data)


def take_word_start_tokens(features, starts_mask, shape=None):
    """
    Maps seq_len tensor to word_seq_len by taking slices from the mask
    Assumes that features.shape[-1] != 0
    :param features: [BATCH_SIZE, SEQ_LEN, ...]
    :param starts_mask: [BATCH_SIZE, SEQ_LEN]
    :param shape: shape of features tensor if needed
    :return: [BATCH_SIZE, SEQ_LEN, ...]
    """
    if shape is None:
        shape = features.get_shape().as_list()

    shape[0] = -1
    shape[1] = tf.math.reduce_max(tf.reduce_sum(starts_mask, -1))
    features = tf.gather_nd(features, tf.where(starts_mask))
    features = tf.reshape(features, shape)
    return features


def word_to_token_level(outputs, word_begins_full_mask):
    """
    Insert zeros
    :param outputs: [BATCH_SIZE, WORD_SEQ_LEN, ...]
    :param word_begins_full_mask: [BATCH_SIZE, SEQ_LEN]
    :return: [BATCH_SIZE, SEQ_LEN]
    """
    word_seq_len = tf.shape(outputs)[1]
    outputs = tf.expand_dims(outputs, axis=1)  # [BATCH_SIZE,  WORD_SEQ_LEN, WORD_SEQ_LEN, ...]

    # create bool matrix [BATCH_SIZE, WORD_SEQ_LEN, SEQ_LEN] such that
    # seq_mask[b, i, j] == 1 <==> i-th token moves to j-th place in b-th sample
    cum_starts_mask = tf.math.cumsum(word_begins_full_mask, 1) * word_begins_full_mask
    cum_starts_mask_shifted = (tf.math.cumsum(word_begins_full_mask, 1) - 1) * word_begins_full_mask
    sm1 = tf.cast(tf.sequence_mask(cum_starts_mask, word_seq_len), outputs.dtype)
    sm2 = tf.cast(tf.sequence_mask(cum_starts_mask_shifted, word_seq_len), outputs.dtype)
    sequence_mask = sm1 - sm2  # [BATCH_SIZE, WORD_SEQ_LEN, SEQ_LEN]

    if len(outputs.get_shape().as_list()) > 3:
        sequence_mask = tf.expand_dims(sequence_mask, -1)

    return tf.math.reduce_sum(outputs * sequence_mask, axis=2)  # [BATCH_SIZE, SEQ_LEN]


def get_padding_length(word_begins_mask, word_seq_len, seq_len):
    """
    In the dataset, sentences are padded to the length of a max token sequence.
    For moving data between SEQ_LEN and WORD_SEQ_LEN tensors, we need to provide
    mapping between word begins position for ALL words, including words that would have been
    added by batch padding. There might not be enough space for these words.
    Example:
    [w1, w1, w1, w1, w2]
    [w3, w4, w5, PAD, PAD]
    WORD_SEQ_LEN=3, so we need 3 tokens be word begins in sentence one, but there is no space for the third token
    (this can be overcome by storing indices) todo
    :param word_begins_mask: [BATCH_SIZE, SEQ_LEN]
    :return: int
    """
    return tf.reduce_max(
        (word_seq_len - tf.range(seq_len - 1, -1, -1) - tf.cumsum(word_begins_mask, axis=1)) * word_begins_mask
    )


def pad_right(data, pad_len):
    shape = data.get_shape().as_list()
    if len(shape) == 2:
        paddings = tf.convert_to_tensor([[0, 0], [0, pad_len]])
    elif len(shape) == 3:
        paddings = tf.convert_to_tensor([[0, 0], [0, pad_len], [0, 0]])
    else:
        raise NotImplementedError
    return tf.pad(data, paddings)


def padded_to_full_word_mask(word_begins_mask, word_seq_len, seq_len):
    """
    Word_begins_mask maps word begin indices to words. Here we add padded words to this mask
    """
    return tf.where(
        tf.greater(
            tf.range(seq_len, 0, -1), word_seq_len - tf.cumsum(word_begins_mask, axis=1)
        ),
        word_begins_mask,
        1
    )
