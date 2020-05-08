import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.compat.v1.logging as logging
import metrics
import output_fns
import util
import transformer_layer
import attention_fns
import constants
import preprocessor_maps
from opennmt.layers.position import SinusoidalPositionEncoder
# https://github.com/OpenNMT/OpenNMT-tf/blob/2c1d81ccd00ff6abd886c180ff81e9821e0fd572/opennmt/layers/position.py#L85


_MAIN_INPUT = 'word_type'


class LISAModel(tf.keras.models.Model):
    # INITIALIZATION PART
    def __init__(self, hparams, model_config, task_config, attention_config,
                 feature_idx_map, label_idx_map, vocab):
        super(LISAModel, self).__init__()
        self.hparams = hparams

        self.model_config = util.NotTrackableDict(model_config)
        self.task_config = util.NotTrackableDict(task_config)
        self.attention_config = util.NotTrackableDict(attention_config)
        self.feature_idx_map = util.NotTrackableDict(feature_idx_map)
        self.label_idx_map = util.NotTrackableDict(label_idx_map)
        self.vocab = vocab
        self.layer_config = self.model_config['layers']

        self.items_to_log = {}
        self.task_list = util.task_list(task_config)

        self.num_layers = max(self.task_config.keys()) + 1

        transition_stats = util.load_transition_params(self.task_config, self.vocab)

        self.embeddings = self.get_embeddings()
        self.init_layers(transition_stats)
        self.init_metrics()
        self.custom_eval = False
        self.teacher_forcing = False
        self.tune_first_layer = False

    def init_layers(self, transition_stats):
        self.initial_dropout = L.Dropout(1 - self.hparams.input_dropout)  # todo AG mb noise_shape=[None, 1, <100>] ?
        self.layer_norms = [L.LayerNormalization() for _ in range(self.num_layers)]
        sa_hidden_size = self.layer_config['head_dim'] * self.layer_config['num_heads']
        self.hparams['sa_hidden_size'] = sa_hidden_size

        if self.model_config['first_layer'] != 'embeddings':
            self.first_layer_model = preprocessor_maps.load_model(self.model_config['first_layer'])

        self.dense1 = L.Dense(sa_hidden_size, activation=L.LeakyReLU(alpha=0.1))
        self.positional_encoder = SinusoidalPositionEncoder()

        self.transformer_layers = []
        for i in range(self.num_layers):
            attn_fns = []
            value_fns = []

            if i in self.attention_config:
                this_layer_attn_config = self.attention_config[i]
                for attn_fn, attn_fn_map in this_layer_attn_config.get('attention_fns', {}).items():
                    attn_fns.append(
                        attention_fns.dispatcher[attn_fn_map['name']](attn_fn_map)
                    )

                for value_fn, value_fn_map in this_layer_attn_config.get('value_fns', {}).items():
                    value_fns.append(
                        attention_fns.dispatcher[value_fn_map['name']](value_fn_map, embeddings=self.embeddings)
                    )
            self.transformer_layers.append(
                transformer_layer.TransformerLayer(i, self.layer_config, attn_fns, value_fns, self.hparams)
            )

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
                    transition_params=transition_stats.get(task),
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
        words = features[_MAIN_INPUT]

        # for masking out padding tokens
        mask = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                        tf.ones([batch_size, batch_seq_len]))
        pred_mask = mask if 'word_begin' not in features else features['word_begin']

        # Extract named features from monolithic "features" input
        features = {f: tf.multiply(tf.cast(mask, tf.int32), v) for f, v in features.items()}
        tokens = features.copy()

        # Extract named labels from monolithic "features" input, and mask them
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

        return features, mask, pred_mask, labels, tokens

    # RUNNING PART
    def compute_masks(self, words, seq_len, word_begins_mask=None):
        """
        :param words: [BATCH_SIZE, SEQ_LEN]
        :param word_begins: [BATCH_SIZE, SEQ_LEN]
        :returns
            token_pad_mask: which tokens are not pad values
            word_begins_mask: which tokens correspond to word begins, excluding pad words
            word_begins_full_mask: which tokens correspond to word begins, including pad words
            word_pad_mask: which words are not pad values
        """
        token_pad_mask = tf.where(tf.equal(words, constants.PAD_VALUE), 0, 1)  # [BATCH_SIZE, SEQ_LEN]

        if word_begins_mask is not None:
            word_begins_mask = tf.where(
                tf.equal(word_begins_mask, 0),
                0, token_pad_mask
            )
            word_seq_len = tf.reduce_max(tf.reduce_sum(word_begins_mask, 1))
            pad_len = util.get_padding_length(word_begins_mask, word_seq_len, seq_len)
            seq_len += pad_len
            token_pad_mask = util.pad_right(token_pad_mask, pad_len)  # [BATCH_SIZE, pad_SEQ_LEN]
            word_begins_mask = util.pad_right(word_begins_mask, pad_len)  # [BATCH_SIZE, pad_SEQ_LEN]
            word_begins_full_mask = util.padded_to_full_word_mask(
                word_begins_mask, word_seq_len, seq_len)  # [BATCH_SIZE, pad_SEQ_LEN]
            word_pad_mask = util.take_word_start_tokens(
                token_pad_mask, word_begins_full_mask)  # [BATCH_SIZE, WORD_SEQ_LEN]
        else:
            pad_len = 0
            word_seq_len = seq_len
            word_begins_mask = token_pad_mask
            word_pad_mask = token_pad_mask
            word_begins_full_mask = tf.ones_like(word_begins_mask)

        masks = {
            'token_pad_mask': token_pad_mask,  # [BATCH_SIZE, pad_SEQ_LEN]
            'word_begins_mask': word_begins_mask,  # [BATCH_SIZE, pad_SEQ_LEN]
            'word_begins_full_mask': word_begins_full_mask,  # [BATCH_SIZE, pad_SEQ_LEN]
            'word_pad_mask': word_pad_mask,  # [BATCH_SIZE, WORD_SEQ_LEN]
        }
        return masks, pad_len, word_seq_len

    def preprocess_batch_push(self, batch):
        batch_shape = tf.shape(batch)
        seq_len = batch_shape[1]

        features = {f: batch[:, :, idx] for f, idx in self.feature_idx_map.items()}

        words = features[_MAIN_INPUT]
        masks, pad_len, word_seq_len = self.compute_masks(words, seq_len, features.get('word_begin'))
        seq_len += pad_len

        # Extract named features from monolithic "features" input
        features = {f: tf.multiply(
            tf.cast(masks['token_pad_mask'], tf.int32), util.pad_right(v, pad_len)
        ) for f, v in features.items()}  # values:[BATCH_SIZE, pad_SEQ_LEN]

        tokens = {
            f: util.take_word_start_tokens(v, masks['word_begins_full_mask']) for f, v in features.items()
        }  # values:[BATCH_SIZE, WORD_LEN]

        # Extract named labels from monolithic "features" input, and mask them
        labels = {}
        token_labels = {}
        for l, idx in self.label_idx_map.items():
            these_labels = batch[:, :, idx[0]:idx[1]] if idx[1] != -1 else batch[:, :, idx[0]:]
            # [BATCH_SIZE, WORD_LEN, label_len]
            these_labels = util.pad_right(these_labels, pad_len)
            this_mask = tf.where(tf.equal(these_labels, constants.PAD_VALUE), 0, 1)
            these_labels_masked = tf.multiply(these_labels, this_mask)
            last_dim = tf.shape(these_labels)[2]

            if idx[1] != -1:
                these_labels_masked = tf.squeeze(these_labels_masked, -1)

            token_labels[l] = these_labels_masked
            word_labels = util.take_word_start_tokens(
                these_labels_masked, masks['word_begins_full_mask'],
                shape=[-1, -1, last_dim] if idx[1] == -1 else None
            )
            labels[l] = word_labels

        return features, masks, labels, token_labels, tokens

    def preprocess_features(self, inputs):
        # Set up model inputs
        features = []
        for input_name in self.model_config['inputs']:  # word type and/or predicate
            if input_name == _MAIN_INPUT and self.model_config['first_layer'] != 'embeddings':
                features.append(
                    self.first_layer_model(inputs[input_name])[0]
                )
            else:
                features.append(
                    tf.nn.embedding_lookup(self.embeddings[input_name], inputs[input_name])
                )

        return tf.concat(features, axis=2)

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
        inputs, masks, labels, token_labels, tokens = self.preprocess_batch_push(batch)
        features = self.preprocess_features(inputs)
        features = tf.stop_gradient(features) if not self.tune_first_layer else features

        outputs = {
            'mask': tf.cast(masks['word_pad_mask'], tf.float32),  # loss needs it
            'tokens': tokens,  # will be used in evaluation, srl.pl needs words  # todo or does it?
        }
        token_level_outputs = {}

        features = self.initial_dropout(features)
        features = self.dense1(features)
        features = self.positional_encoder(features)  # [BATCH_SIZE, SEQ_LEN, SA_HID==NUM_HEADS * HEAD_DIM]

        for i in range(self.num_layers):
            features = self.transformer_layers[i]([features, masks['token_pad_mask'], token_level_outputs, token_labels])

            predict_features = util.take_word_start_tokens(features, masks['word_begins_full_mask'])
            # [BATCH_SIZE, WORD_LEN, SA_HID]

            # if normalization is done in layer_preprocess, then it should also be done
            # on the output, since the output can grow very large, being the sum of
            # a whole stack of unnormalized layer outputs.
            predict_features = self.layer_norms[i](predict_features)

            for task, layer in self.output_layers.items():
                if layer.transformer_layer_id == i:
                    outputs[task] = self.output_layers[task](
                        [predict_features, masks['word_pad_mask']],
                        outputs=outputs,
                        labels=labels,
                    )  # todo doc
                    if i != self.num_layers - 1:
                        # todo AG remove all this duct tape (through configs maybe)
                        token_level_outputs[task] = {
                            k: util.word_to_token_level(v, masks['word_begins_full_mask']) for k, v in
                                outputs[task].items() if k in ['probabilities', 'scores', 'gold_pos_probabilities']
                        }
                        if task == 'parse_head':
                            x = tf.transpose(token_level_outputs[task]['scores'], [0, 2, 1])
                            x = util.word_to_token_level(x, masks['word_begins_full_mask'])
                            x = tf.transpose(x, [0, 2, 1])
                            token_level_outputs[task]['scores'] = x

        losses = self.model_loss(labels, outputs)

        if self.custom_eval:
            self.update_metrics(labels, outputs, losses)

        predictions = self.outputs_to_predictions(outputs)
        if self.custom_eval:
            return [*predictions, tf.zeros_like(masks['word_pad_mask'])]
            # keras wants output_shape[0]==batch_size for all outputs
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

    def enable_teacher_forcing(self):
        self.teacher_forcing = True
        for l in self.transformer_layers:
            l.enable_teacher_forcing()
        for l in self.output_layers.values():
            l.enable_teacher_forcing()

    def disable_teacher_forcing(self):
        for l in self.transformer_layers:
            l.disable_teacher_forcing()
        for l in self.output_layers.values():
            l.disable_teacher_forcing()
        self.teacher_forcing = False

    def start_custom_eval(self):
        self.custom_eval = True
        self.disable_teacher_forcing()
        for f in self.output_layers.values():
            f.in_eval_mode = True
        for metric in self.custom_metrics:
            metric.reset_states()

    def end_custom_eval(self, enable_teacher_forcing=True):
        for f in self.output_layers.values():
            f.in_eval_mode = False
        if enable_teacher_forcing:
            self.enable_teacher_forcing()
        self.custom_eval = False

    def unfreeze_first_layer(self):
        self.tune_first_layer = True
