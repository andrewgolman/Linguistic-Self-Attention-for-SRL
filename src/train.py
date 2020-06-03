import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.optimizers as optim
import tensorflow_addons as tfa

import argparse
import os
import train_utils
from vocab import Vocab
from model import LISAModel
from loss import DummyLoss, SumLoss
import callbacks
import dataset

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
arg_parser.add_argument('--best_eval_key', required=True, type=str,
                        help='Key corresponding to the evaluation to be used for determining early stopping.')
arg_parser.add_argument('--checkpoint', required=False, type=str,
                        help='Start with weights from checkpoint')
arg_parser.add_argument('--disable_teacher_forcing', action="store_true",
                        help='disable_teacher_forcing')
arg_parser.add_argument('--tune_first_layer', action="store_true",
                        help='tune_first_layer')
arg_parser.add_argument('--eval_every', required=True, type=int)
arg_parser.add_argument('--save_every', required=True, type=int)

arg_parser.set_defaults(debug=False)

args, leftovers = arg_parser.parse_known_args()


def main():
    util.init_logging(logging.INFO)

    if not os.path.isdir(args.save_dir):
        util.fatal_error("save_dir not found: %s" % args.save_dir)

    # Load all the various configurations
    data_config = train_utils.load_json_configs(args.data_config)
    model_config = train_utils.load_json_configs(args.model_configs)
    task_config = train_utils.load_json_configs(args.task_configs, args)
    layer_config = train_utils.load_json_configs(args.layer_configs)
    attention_config = train_utils.load_json_configs(args.attention_configs)

    data_config, model_config = util.parse_multifeatures(data_config, model_config)
    # Combine layer, task and layer, attention maps
    # todo save these maps in save_dir
    layer_task_config, layer_attention_config = util.combine_attn_maps(layer_config, attention_config, task_config)

    hparams = train_utils.load_hparams(args, model_config)

    # Set the random seed. This defaults to int(time.time()) if not otherwise set.
    # todo AG check why results differ on similar runs
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

    # Generate mappings from feature/label names to indices in the model_fn inputs
    feature_idx_map, label_idx_map = util.load_feat_label_idx_maps(data_config)

    # todo verify, mb use nadam
    optimizer = optim.Adam(
        learning_rate=hparams.learning_rate,
        beta_1=hparams.beta1,
        beta_2=hparams.beta2,
        epsilon=hparams.epsilon,
        clipnorm=hparams.gradient_clip_norm,
    )

    optimizer = tfa.optimizers.MovingAverage(optimizer, average_decay=hparams.moving_average_decay)
    task_list_size = len(util.task_list(layer_task_config))
    losses = [DummyLoss()] * task_list_size + [SumLoss()]

    # Initialize the model
    # mirrored_strategy = tf.distribute.MirroredStrategy()
    # with mirrored_strategy.scope():
    model = LISAModel(hparams, model_config, layer_task_config, layer_attention_config, feature_idx_map,
                      label_idx_map, vocab)
    model.compile(
        optimizer=optimizer,
        loss=losses,
    )

    lookup_ops = vocab.create_vocab_lookup_ops(embedding_files)
    train_batch_generator = train_utils.batch_generator(
        task_list_size, lookup_ops,
        data_config, train_filenames,
        num_epochs=hparams.train_epochs,
        shuffle=True, batch_size=hparams.batch_size
    )

    val_datasets = [
        dataset.get_dataset(
            [filename], data_config, lookup_ops,
            batch_size=hparams.validation_batch_size, num_epochs=1, shuffle=False
        ) for filename in dev_filenames
    ]

    batch = next(train_batch_generator)
    model.start_custom_eval()
    model(batch[0])
    # do not remove this line, it sets teacher forcing
    model.end_custom_eval(enable_teacher_forcing=not args.disable_teacher_forcing)
    model(batch[0])

    if args.checkpoint:
        # todo AG
        start_epoch = int(args.checkpoint.split("_")[-1])
        logging.log(logging.INFO, "Loading model weights from checkpoint {}".format(args.checkpoint))
        model.load_weights(args.checkpoint)
    else:
        start_epoch = 0

    model.summary()
    # model.fit(train_batch_generator, epochs=1, steps_per_epoch=1)

    if args.tune_first_layer:
        model.unfreeze_first_layer()

    lr_schedule_callback = tf.keras.callbacks.LearningRateScheduler(
        train_utils.learning_rate_scheduler(hparams, start_epoch=start_epoch),
    )
    eval_callbacks = [
        callbacks.EvalMetricsCallBack(
            val_dataset,
            "{}/metrics_log_{}.txt".format(args.save_dir, i),
            eval_every=args.eval_every,
            enable_teacher_forcing=(not args.disable_teacher_forcing)
        )
        for i, val_dataset in enumerate(val_datasets)
    ]
    save_callback = callbacks.SaveCallBack(path=args.save_dir, save_every=args.save_every, start_epoch=start_epoch)

    model.fit(
        train_batch_generator,
        epochs=hparams.train_epochs,
        steps_per_epoch=hparams.steps_per_epoch,
        callbacks=[*eval_callbacks, lr_schedule_callback, save_callback],
    )


if __name__ == "__main__":
    main()
