import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.compat.v1.logging as logging
import metrics
import output_fns
import util
import transformer_layer
import constants
from opennmt.layers.position import SinusoidalPositionEncoder
# https://github.com/OpenNMT/OpenNMT-tf/blob/2c1d81ccd00ff6abd886c180ff81e9821e0fd572/opennmt/layers/position.py#L85
import tensorflow.python.training.tracking.tracking as tracking



# https://github.com/tensorflow/tensorflow/blob/c3973c78f03c50d8514c14c2866ab30e708aea24/tensorflow/python/training/tracking/tracking.py
class NotTrackableDict(tracking.NotTrackable, dict):
    def __init__(self, data):
        tracking.NotTrackable.__init__(self)
        dict.__init__(self, data)


class LISAModel(tf.keras.models.Model):
    # INITIALIZATION PART
    def __init__(self, hparams, model_config, task_config, attention_config,
                 feature_idx_map, label_idx_map, vocab):
        super(LISAModel, self).__init__()
        self.hparams = hparams

        self.model_config = NotTrackableDict(model_config)
        self.task_config = NotTrackableDict(task_config)
        self.attention_config = NotTrackableDict(attention_config)
        self.feature_idx_map = NotTrackableDict(feature_idx_map)
        self.label_idx_map = NotTrackableDict(label_idx_map)
        self.vocab = vocab
        self.layer_config = self.model_config['layers']

        self.items_to_log = {}
        self.task_list = util.task_list(task_config)

        self.num_layers = max(self.task_config.keys()) + 1

        # load transition parameters
        self.transition_stats = util.load_transition_params(self.task_config, self.vocab)
        # todo don't save it

        self.init_layers()
        self.init_metrics()
        self.embeddings = self.get_embeddings()
        self.custom_eval = False
        self.eval_loss_history = []

        logging.log(logging.INFO,
                    "Created model with {} trainable parameters".format(util.count_model_params(self)))

    def init_layers(self):
        self.initial_dropout = L.Dropout(1 - self.hparams.input_dropout)
        self.layer_norm = L.LayerNormalization()  # epsilon=1e-6
        sa_hidden_size = self.layer_config['head_dim'] * self.layer_config['num_heads']
        self.dense1 = L.Dense(sa_hidden_size, activation=L.LeakyReLU(alpha=0.1))

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

    def init_metrics(self):
        self.custom_metrics = [metrics.ValLoss()]
        for layer_id in self.task_config:
            for task, params in self.task_config[layer_id].items():
                for eval_name, eval_map in params['eval_fns'].items():
                    name = eval_map['name']
                    fn = metrics.dispatcher[name](
                        task=task,
                        config=eval_map,
                        reverse_maps=self.vocab.reverse_maps,
                    )
                    self.custom_metrics.append(fn)

    def get_embedding_table(self, name, embedding_dim, include_oov, pretrained_fname=None, num_embeddings=None):
        """
        AG: get table, check dimension, add oov and convert to tensor
        """
        if pretrained_fname:
            pretrained_embeddings = util.load_pretrained_embeddings(pretrained_fname)
            pretrained_num_embeddings, pretrained_embedding_dim = pretrained_embeddings.shape
            if pretrained_embedding_dim != embedding_dim:
                util.fatal_error("Pre-trained %s embedding dim does not match specified dim (%d vs %d)." %
                                 (name, pretrained_embedding_dim, embedding_dim))
            if num_embeddings and num_embeddings != pretrained_num_embeddings:
                util.fatal_error("Number of pre-trained %s embeddings does not match specified "
                                 "number of embeddings (%d vs %d)." % (name, pretrained_num_embeddings, num_embeddings))
            embedding_table = tf.convert_to_tensor(pretrained_embeddings, dtype=tf.float32)
        else:
           embedding_table = tf.random.normal(shape=[num_embeddings, embedding_dim])

        if include_oov:
            oov_embedding = tf.random.normal(shape=[1, embedding_dim])
            embedding_table = tf.concat([embedding_table, oov_embedding], axis=0)

        return embedding_table

    def get_embeddings(self):
        # Create embeddings tables, loading pre-trained if specified
        embeddings = {}
        for embedding_name, embedding_map in self.model_config['embeddings'].items():
            embedding_dim = embedding_map['embedding_dim']
            if 'pretrained_embeddings' in embedding_map:
                input_pretrained_embeddings = embedding_map['pretrained_embeddings']
                include_oov = True
                embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                         pretrained_fname=input_pretrained_embeddings)
            else:
                num_embeddings = self.vocab.vocab_names_sizes[embedding_name]
                include_oov = self.vocab.oovs[embedding_name]
                embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                       num_embeddings=num_embeddings)
            embeddings[embedding_name] = embedding_table
            logging.log(logging.INFO, "Created embeddings for '%s'." % embedding_name)
        return embeddings

    # RUNNING PART
    def preprocess_batch(self, batch):
        batch_shape = tf.shape(batch)
        batch_size = batch_shape[0]
        batch_seq_len = batch_shape[1]

        features = {f: batch[:, :, idx] for f, idx in self.feature_idx_map.items()}

        # ES todo this assumes that word_type is always passed in
        words = features['word_type']

        # for masking out padding tokens
        mask = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                        tf.ones([batch_size, batch_seq_len]))
        # Extract named features from monolithic "features" input
        features = {f: tf.multiply(tf.cast(mask, tf.int32), v) for f, v in features.items()}
        tokens = features.copy()

        # Extract named labels from monolithic "features" input, and mask them
        # ES todo fix masking -- is it even necessary?
        labels = {}
        for l, idx in self.label_idx_map.items():
            these_labels = batch[:, :, idx[0]:idx[1]] if idx[1] != -1 else batch[:, :, idx[0]:]
            these_labels_masked = tf.multiply(these_labels, tf.cast(tf.expand_dims(mask, -1), tf.int32))
            # check if we need to mask another dimension
            if idx[1] == -1:
                last_dim = tf.shape(these_labels)[2]
                this_mask = tf.where(tf.equal(these_labels_masked, constants.PAD_VALUE),
                                     tf.zeros([batch_size, batch_seq_len, last_dim], dtype=tf.int32),
                                     tf.ones([batch_size, batch_seq_len, last_dim], dtype=tf.int32))
                these_labels_masked = tf.multiply(these_labels_masked, this_mask)
            else:
                these_labels_masked = tf.squeeze(these_labels_masked, -1)
            labels[l] = these_labels_masked

        # Set up model inputs
        features = [
            tf.nn.embedding_lookup(self.embeddings[input_name], features[input_name])
            for input_name in self.model_config['inputs']  # word type and/or predicate
        ]
        features = tf.concat(features, axis=2)

        return features, mask, labels, tokens

    def outputs_to_predictions(self, outputs):
        return [
            outputs[task]['predictions'] for task in self.task_list
        ]

    def call(self, batch):
        """
        data:
            features: [BATCH_SIZE, SEQ_LEN, HID]
            mask: [BATCH_SIZE, SEQ_LEN]
            labels: Dict{task : [BATCH_SIZE, SEQ_LEN]} (for srl: [..., 9])
            tokens Dict{word/word_type : [BATCH_SIZE, SEQ_LEN]}
        :return: predictions
        """
        features, mask, labels, tokens = self.preprocess_batch(batch)

        outputs = {
            'mask': mask,  # loss needs it
            'tokens': tokens,  # todo AG how come we need it
        }

        features = self.initial_dropout(features)
        features = self.dense1(features)
        features = self.positional_encoder(features)  # [BATCH_SIZE, SEQ_LEN, SA_HID==NUM_HEADS * HEAD_DIM]

        for i in range(self.num_layers):
            features = self.transformer_layers[i]([features, mask, outputs, labels])

            # if normalization is done in layer_preprocess, then it should also be done
            # on the output, since the output can grow very large, being the sum of
            # a whole stack of unnormalized layer outputs.
            predict_features = self.layer_norm(features)

            for task, layer in self.output_layers.items():
                if layer.transformer_layer_id == i:
                    outputs[task] = self.output_layers[task](
                        [predict_features, mask],
                        outputs=outputs,
                        labels=labels,
                    )  # todo doc

        losses = self.model_loss(labels, outputs)

        if self.custom_eval:
            self.update_metrics(labels, outputs, losses)

        predictions = self.outputs_to_predictions(outputs)
        if self.custom_eval:
            return [*predictions, tf.zeros_like(mask)]  # keras wants output_shape[0]==batch_size for all outputs
        else:
            return [*predictions, tf.convert_to_tensor(losses, dtype=tf.float32)]

    # LOSS PART
    def model_loss(self, labels, predictions):
        loss = []

        for task_data in self.task_config.values():
            for task, task_map in task_data.items():
                this_output_layer = self.output_layers[task]
                this_task_loss = this_output_layer.loss(labels[task], predictions[task], mask=predictions['mask'])

                loss.append(
                    this_task_loss * task_map['penalty']
                )

        return loss

    # EVALUATION PART
    def update_metrics(self, labels, outputs, losses):
        self.custom_metrics[0].update(None, losses)
        for metric in self.custom_metrics[1:]:
            metric.update(labels, outputs)

    def get_metrics(self):
        scores = {}
        for metric in self.custom_metrics:
            scores[(metric.task, metric.name)] = metric.result()
        return scores

    def start_custom_eval(self):
        self.custom_eval = True
        self.eval_loss_history = []
        for f in self.output_layers.values():
            f.in_eval_mode = True
        for metric in self.custom_metrics:
            metric.reset_states()

    def end_custom_eval(self):
        for f in self.output_layers.values():
            f.in_eval_mode = False
        self.custom_eval = False
