import tensorflow as tf
import tensorflow.keras.optimizers as optim
import tensorflow_addons as tfa
import tensorflow.compat.v1.logging as logging
from tqdm import tqdm
import numpy as np
import os

import argparse
import train_utils
from vocab import Vocab
from model import LISAModel
from loss import DummyLoss, SumLoss
import dataset
import util
from callbacks import print_model_metrics

arg_parser = argparse.ArgumentParser(description='')
arg_parser.add_argument('--data', required=True,
                        help='Path to a file with data paths')
arg_parser.add_argument('--config', required=True,
                        help="Path to a file with config mappings")
arg_parser.add_argument('--save_dir', required=True,
                        help='Directory to save models, outputs, etc.')
arg_parser.add_argument('--hparams', type=str,
                        help='Comma separated list of "name=value" hyperparameter settings.')
arg_parser.add_argument('--checkpoint', required=False, type=str,
                        help='Start with weights from a checkpoint')

args, leftovers = arg_parser.parse_known_args()


def main():
    util.init_logging(logging.INFO)

    data_paths = train_utils.load_data_config(args.data)
    global_config = train_utils.load_global_config(args.config)

    # Load all the various configurations
    data_config = train_utils.load_json_configs(global_config['data_configs'])
    model_config = train_utils.load_json_configs(global_config['model_configs'])
    layer_config = train_utils.load_json_configs(global_config['layer_configs'])
    attention_config = train_utils.load_json_configs(global_config['attention_configs'])
    task_config = train_utils.load_json_configs(global_config['task_configs'], **vars(args), **data_paths)

    data_config, model_config = util.parse_multifeatures(data_config, model_config)
    layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

    hparams = train_utils.load_hparams(args, model_config)
    np.random.seed(hparams.random_seed)
    tf.random.set_seed(hparams.random_seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    vocab = Vocab(data_config, args.save_dir, data_paths['train'])
    vocab.update_vocab_files(data_paths['dev'])

    # Generate mappings from feature/label names to indices in the model_fn inputs
    feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)

    optimizer = optim.Adam(
        learning_rate=1e-10,
        clipnorm=hparams.gradient_clip_norm,
    )
    optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=hparams.moving_average_decay)
    task_list_size = len(util.task_list(layer_task_config))
    losses = [DummyLoss()] * task_list_size + [SumLoss()]

    embedding_files = [embeddings_map['pretrained_embeddings'] for embeddings_map in model_config['embeddings'].values()
                       if 'pretrained_embeddings' in embeddings_map]
    lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)

    test_dataset = dataset.get_dataset(data_paths['test'], data_config, lookup_ops, batch_size=128, num_epochs=1, shuffle=False)
    dev_dataset = dataset.get_dataset(data_paths['dev'], data_config, lookup_ops, batch_size=128, num_epochs=1, shuffle=False)

    model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map,
                      label_idx_map, vocab)
    model.compile(optimizer=optimizer, loss=losses)

    train_batch_generator = train_utils.batch_generator(
        task_list_size, lookup_ops,
        data_config, data_paths['dev'],
        num_epochs=1,
        shuffle=False, batch_size=8
    )
    batch = next(train_batch_generator)
    model.start_custom_eval()
    model(batch[0])
    model.end_custom_eval()

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
