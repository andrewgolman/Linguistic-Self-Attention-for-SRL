import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optim

import argparse
import os
from functools import partial
import train_utils
from vocab import Vocab
from model import LISAModel, LISAModelLoss
import numpy as np
import util
import tensorflow.compat.v1.logging as logging


arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_files', required=True,
                        help='Comma-separated list of training data files')
arg_parser.add_argument('--dev_files', required=True,
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory to save models, outputs, etc.')
# todo load this more generically, so that we can have diff stats per task
arg_parser.add_argument('--transition_stats',
                        help='Transition statistics between labels')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
arg_parser.add_argument('--debug', dest='debug', action='store_true',
                        help='Whether to run in debug mode: a little faster and smaller')
arg_parser.add_argument('--data_config', required=True,
                        help='Path to data configuration json')
arg_parser.add_argument('--model_configs', required=True,
                        help='Comma-separated list of paths to model configuration json.')
arg_parser.add_argument('--task_configs', required=True,
                        help='Comma-separated list of paths to task configuration json.')
arg_parser.add_argument('--layer_configs', required=True,
                        help='Comma-separated list of paths to layer configuration json.')
arg_parser.add_argument('--attention_configs',
                        help='Comma-separated list of paths to attention configuration json.')
arg_parser.add_argument('--num_gpus', type=int,
                        help='Number of GPUs for distributed training.')
arg_parser.add_argument('--keep_k_best_models', type=int,
                        help='Number of best models to keep.')
arg_parser.add_argument('--best_eval_key', required=True, type=str,
                        help='Key corresponding to the evaluation to be used for determining early stopping.')

arg_parser.set_defaults(debug=False, num_gpus=1, keep_k_best_models=1)

args, leftovers = arg_parser.parse_known_args()

util.init_logging(logging.INFO)

# Load all the various configurations
# todo: validate json
data_config = train_utils.load_json_configs(args.data_config)
model_config = train_utils.load_json_configs(args.model_configs)
task_config = train_utils.load_json_configs(args.task_configs, args)
layer_config = train_utils.load_json_configs(args.layer_configs)
attention_config = train_utils.load_json_configs(args.attention_configs)

# attention_config = {}
# if args.attention_configs and args.attention_configs != '':
#   attention_config = train_utils.load_json_configs(args.attention_configs)

# Combine layer, task and layer, attention maps
# todo save these maps in save_dir
layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

hparams = train_utils.load_hparams(args, model_config)

# Set the random seed. This defaults to int(time.time()) if not otherwise set.
np.random.seed(hparams.random_seed)
tf.random.set_seed(hparams.random_seed)

if not os.path.exists(args.save_dir):
  os.makedirs(args.save_dir)

train_filenames = args.train_files.split(',')
dev_filenames = args.dev_files.split(',')

vocab = Vocab(data_config, args.save_dir, train_filenames)
vocab.update_vocab_files(dev_filenames)

embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                   if 'pretrained_embeddings' in embeddings_map]


def dev_batch_generator(preprocessor):
    # return train_utils.get_input_fn(vocab, data_config, dev_filenames, hparams.batch_size, num_epochs=1, shuffle=False,
    #                                 embedding_files=embedding_files)
    pass


def main():
    # Generate mappings from feature/label names to indices in the model_fn inputs
    feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)

    # Initialize the model
    model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map, label_idx_map,
                      vocab)

    if args.debug:
      logging.log(logging.INFO, "Created trainable variables: %s" % str([v.name for v in tf.trainable_variables()]))

    preprocessor = model.get_preprocessor_instance()
    optimizer = optim.Adam(
        # learning_rate=this_step_lr,
        beta_1=hparams.beta1,
        beta_2=hparams.beta2,
        epsilon=hparams.epsilon,
        # clipnorm=hparams.gradient_clip_norm,  # todo AG
    )
    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(train_utils.learning_rate_scheduler(hparams))

    model.compile(
        optimizer=optimizer,
        loss=LISAModelLoss(model.task_config, model.vocab)
    )
    batch_generator = train_utils.train_batch_generator(preprocessor,
                                          vocab, data_config, dev_filenames, num_epochs=1,
                                          shuffle=False,
                                          embedding_files=embedding_files,
                                          batch_size=2)  # hparams.batch_size
    batch, labels = next(batch_generator)
    model(batch)

    # model.fit_generator(  # todo
    #
    #     epochs=5,
    #     steps_per_epoch=100,
    #     callbacks=[lr_schedule_callback],
    # )


if __name__ == "__main__":
    main()
