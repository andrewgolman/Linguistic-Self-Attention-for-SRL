import tensorflow as tf
import json
import re
import sys
import dataset
import constants
from pathlib import Path
import tensorflow.compat.v1.logging as logging
from attrdict import AttrDict


def load_hparams(args, model_config):
    # Create a HParams object specifying the names and values of the model hyperparameters
    hparams = AttrDict(**constants.hparams)

    # First get default hyperparams from the model config
    hparams.update(model_config.get('hparams', {}))

    # Override those with command line hyperparams
    if args.hparams:
        hparams.update(vars(args.hparams))

    logging.log(logging.INFO, "Using hyperparameters: {}".format(hparams))
    return hparams


def batch_generator(task_count, lookup_ops, data_config, data_files, batch_size, num_epochs, shuffle,
                    shuffle_buffer_multiplier=1, repeat=True):
    # todo AG check graph for lookup ops
    # vocab ops needs to be created from here (lazily) so that it ends up in the same tf.Graph as everything else
    ds = dataset.get_dataset(data_files, data_config, lookup_ops, batch_size, num_epochs, shuffle,
                             shuffle_buffer_multiplier)
    for batch in ds.as_numpy_iterator():
        yield batch, [0] * (task_count + 1)
    if repeat:
        while True:
            for batch in ds.as_numpy_iterator():
                yield batch, [0] * (task_count + 1)


required_configs = {
    'data_configs',
    'model_configs',
    'layer_configs',
    'task_configs',
    'attention_configs',
}


def load_global_config(path):
    with open(path) as f:
        try:
            config = json.load(f)
            for c in required_configs:
                assert isinstance(config.get(c), list)
            return config

        except json.decoder.JSONDecodeError as e:
            logging.log(logging.ERROR, 'Error reading json: "%s"' % path)
            logging.log(logging.ERROR, e.msg)
            sys.exit(1)


def load_data_config(path):
    with open(path) as f:
        try:
            return json.load(f)
        except json.decoder.JSONDecodeError as e:
            logging.log(logging.ERROR, 'Error reading json: "%s"' % path)
            logging.log(logging.ERROR, e.msg)
            sys.exit(1)


def load_json_configs(config_files, **args):
  """
  Loads a list of json configuration files into one combined map. Configuration files
  at the end of the list take precedence over earlier configuration files (so they will
  overwrite earlier configs!)

  If args is passed, then this function will attempt to replace entries surrounded with
  the special tokens ## ## with an entry from args with the same name.

  :param config_files: list of json configuration files to load
  :param args: command line args to replace special strings in json
  :return: map containing combined configurations
  """
  combined_config = {}
  for config_file in config_files:
      if args:
        # read the json in as a string so that we can run a replace on it
        json_str = Path(config_file).read_text()
        matches = re.findall(r'.*##(.*)##.*', json_str)
        for match in matches:
          try:
            value = args[match]
            json_str = json_str.replace('##%s##' % match, value)
          except AttributeError:
            logging.log(logging.ERROR, 'Could not find "%s" attribute in command line args when parsing: %s' %
                           (match, config_file))
            sys.exit(1)
        try:
          config = json.loads(json_str)
        except json.decoder.JSONDecodeError as e:
          logging.log(logging.ERROR, 'Error reading json: "%s"' % config_file)
          logging.log(logging.ERROR, e.msg)
          sys.exit(1)
      else:
        with open(config_file) as f:
          try:
            config = json.load(f)
          except json.decoder.JSONDecodeError as e:
            logging.log(logging.ERROR, 'Error reading json: "%s"' % config_file)
            logging.log(logging.ERROR, e.msg)
            sys.exit(1)
      combined_config = {**combined_config, **config}
  return combined_config


def learning_rate(hparams, global_step):
    lr = hparams.learning_rate
    warmup_steps = hparams.warmup_steps
    decay_rate = hparams.decay_rate
    if warmup_steps > 0:
        # add 1 to global_step so that we start at 1 instead of 0
        global_step_float = tf.cast(global_step, tf.float32) + 1.
        lr *= tf.minimum(tf.math.rsqrt(global_step_float),
                         tf.multiply(global_step_float, warmup_steps ** -decay_rate))
        return lr
    else:
        decay_steps = hparams.decay_steps
        if decay_steps > 0:
            return lr * decay_rate ** (global_step / decay_steps)
        else:
            return lr


def learning_rate_scheduler(hparams, start_epoch=None):
    steps_per_epoch = hparams.steps_per_epoch

    if start_epoch is None:
        start_epoch = 0

    def callback(epoch):
        return learning_rate(hparams, (epoch + start_epoch) * steps_per_epoch).numpy()

    return callback
