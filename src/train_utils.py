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

    if args.debug:
        hparams['shuffle_buffer_multiplier'] = 10
        hparams['eval_throttle_secs'] = 60
        hparams['eval_every_steps'] = 100

    # Override those with command line hyperparams
    if args.hparams:
        hparams.update(vars(args.hparams))

    logging.log(logging.INFO, "Using hyperparameters: {}".format(hparams.values()))
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


def load_json_configs(config_file_list, args=None):
  """
  Loads a list of json configuration files into one combined map. Configuration files
  at the end of the list take precedence over earlier configuration files (so they will
  overwrite earlier configs!)

  If args is passed, then this function will attempt to replace entries surrounded with
  the special tokens ## ## with an entry from args with the same name.

  :param config_file_list: list of json configuration files to load
  :param args: command line args to replace special strings in json
  :return: map containing combined configurations
  """
  combined_config = {}
  if config_file_list:
    config_files = config_file_list.split(',')
    for config_file in config_files:
      if args:
        # read the json in as a string so that we can run a replace on it
        json_str = Path(config_file).read_text()
        matches = re.findall(r'.*##(.*)##.*', json_str)
        for match in matches:
          try:
            value = getattr(args, match)
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


def get_vars_for_moving_average(average_norms):
  vars_to_average = tf.trainable_variables()
  if not average_norms:
    vars_to_average = [v for v in tf.trainable_variables() if 'norm' not in v.name]
  logging.log(logging.INFO, "Creating moving averages for %d variables." % len(vars_to_average))
  return vars_to_average


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


def learning_rate_scheduler(hparams):
    steps_per_epoch = hparams.steps_per_epoch

    def callback(epoch):
        return learning_rate(hparams, epoch * steps_per_epoch).numpy()

    return callback


def best_model_compare_fn(best_eval_result, current_eval_result, key):
  """Compares two evaluation results and returns true if the second one is greater.
    Both evaluation results should have the value for key, used for comparison.
    Args:
      best_eval_result: best eval metrics.
      current_eval_result: current eval metrics.
      key: key to value used for comparison.
    Returns:
      True if the loss of current_eval_result is smaller; otherwise, False.
    Raises:
      ValueError: If input eval result is None or no loss is available.
    """

  if not best_eval_result or key not in best_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  if not current_eval_result or key not in current_eval_result:
    raise ValueError('best_eval_result cannot be empty or key "%s" is not found.' % key)

  return best_eval_result[key] < current_eval_result[key]
