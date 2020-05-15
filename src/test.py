import tensorflow.keras.optimizers as optim
import tensorflow.compat.v1.logging as logging

import tensorflow_addons as tfa
import argparse
import os
from tqdm import tqdm

import dataset
import train_utils
import util

from model import LISAModel
from vocab import Vocab
from loss import DummyLoss, SumLoss
from callbacks import print_model_metrics


arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--train_files', required=True,
                        help='Comma-separated list of training data files')
arg_parser.add_argument('--dev_files', required=True,
                        help='Comma-separated list of development data files')
arg_parser.add_argument('--test_files', required=True,
                        help='Comma-separated list of test data files')
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory to save models, outputs, etc.')
arg_parser.add_argument('--checkpoint', required=True,
                        help='Saved model')
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

from tensorflow.keras.backend import manual_variable_initialization
manual_variable_initialization(True)


def main():
    util.init_logging(logging.INFO)

    # Load all the various configurations
    data_config = train_utils.load_json_configs(args.data_config)
    model_config = train_utils.load_json_configs(args.model_configs)
    task_config = train_utils.load_json_configs(args.task_configs, args)
    layer_config = train_utils.load_json_configs(args.layer_configs)
    attention_config = train_utils.load_json_configs(args.attention_configs)

    # Combine layer, task and layer, attention maps
    layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

    hparams = train_utils.load_hparams(args, model_config)

    train_filenames = args.train_files.split(',')
    dev_filenames = args.dev_files.split(',')
    test_filenames = args.test_files.split(',')

    vocab = Vocab(data_config, args.save_dir, train_filenames)
    vocab.update_vocab_files(dev_filenames)
    # todo AG we might want to handle vocabs differently
    vocab.update_vocab_files(test_filenames)

    embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                       if 'pretrained_embeddings' in embeddings_map]

    # Generate mappings from feature/label names to indices in the model_fn inputs
    feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)

    optimizer = optim.Adam(
        learning_rate=1e-10,
        clipnorm=hparams.gradient_clip_norm,
    )
    optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=hparams.moving_average_decay)
    task_list_size = len(util.task_list(layer_task_config))
    losses = [DummyLoss()] * task_list_size + [SumLoss()]

    lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)

    test_dataset = dataset.get_dataset(test_filenames, data_config, lookup_ops, batch_size=128, num_epochs=1, shuffle=False)
    dev_dataset = dataset.get_dataset(dev_filenames, data_config, lookup_ops, batch_size=128, num_epochs=1, shuffle=False)

    model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map,
                      label_idx_map, vocab)
    model.compile(optimizer=optimizer, loss=losses)

    train_batch_generator = train_utils.batch_generator(
        task_list_size, lookup_ops,
        data_config, dev_filenames,
        num_epochs=1,
        shuffle=False, batch_size=8
    )
    batch = next(train_batch_generator)
    model.start_custom_eval()
    model(batch[0])

    model.end_custom_eval()
    # model.fit(train_batch_generator, epochs=1, steps_per_epoch=1)
    model.load_weights(args.checkpoint)

    model.start_custom_eval()
    for batch in tqdm(test_dataset.as_numpy_iterator()):
        model(batch)

    print("TEST DATASET:")
    metrics = model.get_metrics()
    print_model_metrics(metrics)

    model.start_custom_eval()
    for batch in tqdm(dev_dataset.as_numpy_iterator()):
        model(batch)

    print("DEV DATASET:")
    metrics = model.get_metrics()
    print_model_metrics(metrics)


if __name__ == "__main__":
    main()
