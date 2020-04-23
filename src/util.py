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
