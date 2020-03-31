import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import evaluation_fns
import output_fns
import nn_utils
import train_utils
import tf_utils
import util
import tensorflow.compat.v1.logging as logging
from preprocess_batch import LISAModelPreprocess
import transformer_layer
from opennmt.layers.position import SinusoidalPositionEncoder
# https://github.com/OpenNMT/OpenNMT-tf/blob/2c1d81ccd00ff6abd886c180ff81e9821e0fd572/opennmt/layers/position.py#L85


class LISAModel(keras.models.Model):

    def __init__(self, hparams, model_config, task_config, attention_config, feature_idx_map, label_idx_map,
                 vocab):
        super(LISAModel, self).__init__()
        self.hparams = hparams

        self.model_config = model_config
        self.task_config = task_config
        self.attention_config = attention_config
        self.feature_idx_map = feature_idx_map
        self.label_idx_map = label_idx_map
        self.vocab = vocab
        self.layer_config = self.model_config['layers']

        self.items_to_log = {}

        # число слоев в конфиге
        # для каждого слоя есть конфиг
        self.num_layers = max(self.task_config.keys()) + 1

        self.init_layers()
        # load transition parameters
        self.transition_stats = util.load_transition_params(self.task_config, self.vocab)
        # todo не сохранять это

    def init_layers(self):
        self.initial_dropout = L.Dropout(self.hparams.input_dropout)
        self.layer_norm = L.LayerNormalization(epsilon=1e-6)
        sa_hidden_size = self.layer_config['head_dim'] * self.layer_config['num_heads']
        self.dense1 = L.Dense(sa_hidden_size)

        self.transformer_layers = [
            transformer_layer.TransformerLayer(i+1, self.task_config.get(i), self.layer_config, self.hparams)
            for i in range(self.num_layers)
        ]

        self.positional_encoder = SinusoidalPositionEncoder()

        # transition_params = None
        #
        # if task == 'srl_bilinear':
        #     # Set up CRF / Viterbi transition params if specified
        #     # transition_stats_file = task_map['transition_stats'] if 'transition_stats' in task_map else None
        #     task_transition_stats = self.transition_stats[task] if task in self.transition_stats else None
        #
        #     # create transition parameters if training or decoding with crf/viterbi
        #     if task_map.get('viterbi') or task_map.get('crf'):
        #         transition_params = tf.convert_to_tensor(task_transition_stats)

        self.output_layers = {}
        for layer_id in self.task_config:
            for task, params in self.task_config[layer_id].items():
                task_vocab_size = self.vocab.vocab_names_sizes[task] if task in self.vocab.vocab_names_sizes else -1

                self.output_layers[task] = output_fns.dispatcher[task](
                    transformer_layer_id=layer_id,
                    task_map=params['output_fn'],
                    model_config=self.model_config,
                    task_vocab_size=task_vocab_size,
                    joint_lookup_maps=self.vocab.joint_label_lookup_maps,
                    transition_params=None,
                    hparams=self.hparams,
                    **params
                )

    def call(self, data):
        features, mask = data
        features = self.initial_dropout(features)
        features = self.dense1(features)

        predictions = {}

        features = self.positional_encoder(features)
        for i in range(self.num_layers):
            features = self.transformer_layer[i](features, mask, predictions)

            # if normalization is done in layer_preprocess, then it should also be done
            # on the output, since the output can grow very large, being the sum of
            # a whole stack of unnormalized layer outputs.
            predict_features = self.layer_norm(features)

            for task, layer in self.output_layers:
                if layer.transformer_layer_id == i:
                    predictions[task] = self.get_task_output(task, predict_features, mask)

        # return features
        return predictions

    def get_task_output(self, task, features, mask):
        return self.output_layers[task]([features, mask])

    def get_preprocessor_instance(self):
        return LISAModelPreprocess(self.feature_idx_map, self.label_idx_map, self.model_config, self.vocab)

    def get_loss_instance(self):
        return LISAModelLoss(self.output_layers, self.task_config)


class LISAModelLoss:
    def __init__(self, output_layers, task_config):
        self.items_to_log = {}
        self.output_layers = output_layers
        self.task_config = task_config

    def __call__(self, labels, predictions):
        loss = 0

        for task_data in self.task_config.values():
            for task, task_map in task_data.items():
                this_output_layer = self.output_layers[task]
                this_task_loss = this_output_layer.loss(labels[task], predictions[task])

                self.items_to_log['{}_loss'.format(task)] = this_task_loss
                loss += this_task_loss * task_map['penalty']

                # do the evaluation
                # for eval_name, eval_map in task_map['eval_fns'].items():
                #     eval_fn_params = evaluation_fns.get_params(task_outputs, eval_map, predictions, feats, labels,
                #                                                task_labels, self.vocab.reverse_maps, mask)
                    # eval_result = evaluation_fns.dispatch(eval_map['name'])(**eval_fn_params)
                    # eval_metric_ops[eval_name] = eval_result

        return loss


# def train_step(self):
#     if self.hparams.moving_average_decay > 0.:
#         moving_averager = tf.train.ExponentialMovingAverage(self.hparams.moving_average_decay, zero_debias=True,
#                                                           num_updates=tf.train.get_global_step())
#         moving_average_op = moving_averager.apply(train_utils.get_vars_for_moving_average(self.hparams.average_norms))
#         # use moving averages of variables if evaluating
#         assign_moving_averages_dep = tf.cond(tf.equal(self.mode, ModeKeys.TRAIN),
#                                            lambda: tf.no_op(),
#                                            lambda: nn_utils.set_vars_to_moving_average(moving_averager))
#         # todo AG adjust logging
#         # items_to_log['loss'] = loss
#         # logging_hook = tf.train.LoggingTensorHook(self.items_to_log, every_n_iter=20)
#
#         export_outputs = {tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
#                           tf.estimator.export.PredictOutput(flat_predictions)}
#         logging.log(logging.INFO,
#                   "Created model with %d trainable parameters" % tf_utils.get_num_trainable_parameters())
