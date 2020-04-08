import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as L
import tensorflow.compat.v1.logging as logging
import evaluation_fns
import output_fns
import util
import transformer_layer
from preprocess_batch import LISAModelPreprocess
from opennmt.layers.position import SinusoidalPositionEncoder
# https://github.com/OpenNMT/OpenNMT-tf/blob/2c1d81ccd00ff6abd886c180ff81e9821e0fd572/opennmt/layers/position.py#L85


class LISAModel(keras.models.Model):

    def __init__(self, hparams, model_config, task_config, attention_config,
                 feature_idx_map, label_idx_map, vocab):
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

        self.num_layers = max(self.task_config.keys()) + 1

        # load transition parameters
        self.transition_stats = util.load_transition_params(self.task_config, self.vocab)
        # todo don't save it

        self.init_layers()

        logging.log(logging.INFO,
                    "Created model with {} trainable parameters".format(util.count_model_params(self)))

    def init_layers(self):
        self.initial_dropout = L.Dropout(self.hparams.input_dropout)
        self.layer_norm = L.LayerNormalization(epsilon=1e-6)
        sa_hidden_size = self.layer_config['head_dim'] * self.layer_config['num_heads']
        self.dense1 = L.Dense(sa_hidden_size)

        self.transformer_layers = [
            transformer_layer.TransformerLayer(i+1, self.task_config.get(i),
                                               self.layer_config, self.hparams, self.attention_config)
            for i in range(self.num_layers)
        ]
        self.positional_encoder = SinusoidalPositionEncoder()

        self.output_layers = {}
        for layer_id in self.task_config:
            for task, params in self.task_config[layer_id].items():
                task_vocab_size = self.vocab.vocab_names_sizes[task] if task in self.vocab.vocab_names_sizes else -1

                self.output_layers[task] = output_fns.dispatcher[params['output_fn']['name']](
                    transformer_layer_id=layer_id,
                    task_map=params['output_fn'],
                    model_config=self.model_config,
                    task_vocab_size=task_vocab_size,
                    joint_lookup_maps=self.vocab.joint_label_lookup_maps,
                    transition_params=self.transition_stats.get(task),
                    hparams=self.hparams,
                )

    def get_metrics(self):
        metrics = []
        for layer_id in self.task_config:
            for task, params in self.task_config[layer_id].items():
                for eval_name, eval_map in params['eval_fns'].items():
                    name = eval_map['name']
                    fn = evaluation_fns.dispatcher[name](
                        task=task,
                        config=eval_map,
                        reverse_maps=self.vocab.reverse_maps,
                    )
                    metrics.append(fn)

        return metrics

    def call(self, data):
        """
        :param data:
            features: [BATCH_SIZE, SEQ_LEN, HID]
            mask: [BATCH_SIZE, SEQ_LEN]
            labels: Dict{task : [BATCH_SIZE, SEQ_LEN]} (for srl: [..., 9])
            data
        :return: features, tokens, mask, predictions
        """
        features, mask, labels, tokens = data

        predictions = {
            'mask': mask,  # loss needs it
            'tokens': tokens  # todo AG how come we need it
        }

        features = self.initial_dropout(features)
        features = self.dense1(features)
        features = self.positional_encoder(features)  # [BATCH_SIZE, SEQ_LEN, SA_HID==NUM_HEADS * HEAD_DIM]

        for i in range(self.num_layers):
            features = self.transformer_layers[i]([features, mask, predictions, labels])

            # if normalization is done in layer_preprocess, then it should also be done
            # on the output, since the output can grow very large, being the sum of
            # a whole stack of unnormalized layer outputs.
            predict_features = self.layer_norm(features)

            for task, layer in self.output_layers.items():
                if layer.transformer_layer_id == i:
                    predictions[task] = self.output_layers[task](
                        [predict_features, mask],
                        outputs=predictions,
                        labels=labels,
                    )  # todo doc

        return features, predictions

    def get_preprocessor_instance(self):
        return LISAModelPreprocess(self.feature_idx_map, self.label_idx_map, self.model_config, self.vocab)


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

