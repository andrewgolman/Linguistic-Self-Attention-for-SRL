import tensorflow as tf
import nn_utils
import tensorflow.keras.layers as L
from base_fns import FunctionDispatcher
from tensorflow_addons.text.crf import crf_decode, crf_log_likelihood


class OutputLayer(FunctionDispatcher):
    def __init__(self, transformer_layer_id, task_map, **params):
        super(OutputLayer, self).__init__(task_map, **params)
        self.transformer_layer_id = transformer_layer_id
        self.hparams = params['hparams']
        if task_map.get('viterbi') or task_map.get('crf'):
            self.static_params['transition_params'] = tf.convert_to_tensor(self.static_params['transition_params'])
        else:
            self.static_params['transition_params'] = None

    def make_call(self, data, **params):
      raise NotImplementedError

    def loss(self, targets, output, mask):
      raise NotImplementedError


class SoftmaxClassifier(OutputLayer):
    def __init__(self, transformer_layer_id, task_vocab_size, **params):
        super(SoftmaxClassifier, self).__init__(transformer_layer_id, **params)
        self.dropout = L.Dropout(1 - self.hparams.mlp_dropout)
        self.dense = L.Dense(task_vocab_size)

    def make_call(self, data, **kwargs):
        features, mask = data
        features = self.dropout(features)
        logits = self.dense(features)

        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        output = {
            'predictions': predictions,
            'scores': logits,
            'probabilities': tf.nn.softmax(logits, -1)
        }
        return output

    def loss(self, targets, output, mask):
        logits = output['scores']
        n_labels = self.static_params['task_vocab_size']
        targets_onehot = tf.one_hot(indices=targets, depth=n_labels, axis=-1)
        return tf.compat.v1.losses.softmax_cross_entropy(logits=tf.reshape(logits, [-1, n_labels]),
                                               onehot_labels=tf.reshape(targets_onehot, [-1, n_labels]),
                                               weights=tf.reshape(mask, [-1]),
                                               label_smoothing=self.hparams.label_smoothing,
                                               reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

        # return losses.sparse_categorical_cross_entropy(y_true, y_pred)


class JointSoftmaxClassifier(OutputLayer):
    def __init__(self, transformer_layer_id, **params):
        super(JointSoftmaxClassifier, self).__init__(transformer_layer_id, **params)
        self.dense1 = L.Dense(self.static_params['model_config']['predicate_pred_mlp_size'])
        self.dropout1 = L.Dropout(1 - self.hparams.mlp_dropout)
        self.dense2 = L.Dense(self.static_params['task_vocab_size'])
        self.dropout2 = L.Dropout(1 - self.hparams.mlp_dropout)

    def get_separate_scores_preds_from_joint(self, joint_outputs, joint_num_labels):
      joint_maps = self.static_params['joint_maps']
      predictions = joint_outputs['predictions']
      scores = joint_outputs['scores']
      output_shape = tf.shape(predictions)
      batch_size = output_shape[0]
      batch_seq_len = output_shape[1]
      sep_outputs = {}
      for map_name, label_comp_map in joint_maps.items():
            short_map_name = map_name.split('_to_')[-1]

            # marginalize out probabilities for this task
            task_num_labels = tf.shape(tf.unique(tf.reshape(label_comp_map, [-1]))[0])[0]
            joint_probabilities = tf.nn.softmax(scores)
            joint_probabilities_flat = tf.reshape(joint_probabilities, [-1, joint_num_labels])
            segment_ids = tf.squeeze(tf.nn.embedding_lookup(label_comp_map, tf.range(joint_num_labels)), -1)
            segment_scores = tf.math.unsorted_segment_sum(tf.transpose(joint_probabilities_flat), segment_ids, task_num_labels)
            segment_scores = tf.reshape(tf.transpose(segment_scores), [batch_size, batch_seq_len, task_num_labels])
            sep_outputs["%s_probabilities" % short_map_name] = segment_scores

            # use marginalized probabilities to get predictions
            sep_outputs["%s_predictions" % short_map_name] = tf.argmax(segment_scores, -1)
      return sep_outputs

    def make_call(self, data, **kwargs):
        # features: [BATCH_SIZE, SEQ_LEN, SA_HID]
        # mask: [BATCH_SIZE, SEQ_LEN]
        features, mask = data
        logits = self.dropout1(features)
        logits = self.dense1(logits)
        logits = self.dropout2(logits)
        logits = self.dense2(logits)

        predictions = tf.cast(tf.argmax(logits, axis=-1), tf.int32)

        output = {
            'predictions': predictions,  # [BATCH_SIZE, SEQ_LEN]
            'scores': logits,  # [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
            'probabilities': tf.nn.softmax(logits, -1)  # [BATCH_SIZE, SEQ_LEN, VOCAB_SIZE]
        }

        n_labels = self.static_params['task_vocab_size']
        # now get separate-task scores and predictions for each of the maps we've passed through
        separate_output = self.get_separate_scores_preds_from_joint(output, n_labels)
        output.update(separate_output)

        return output

    def loss(self, targets, output, mask):
        logits = output['scores']
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        n_tokens = tf.reduce_sum(mask)
        return tf.reduce_sum(cross_entropy * mask) / n_tokens


class ParseBilinear(OutputLayer):
    def __init__(self, transformer_layer_id, **params):
        super(ParseBilinear, self).__init__(transformer_layer_id, **params)
        self.class_mlp_size = self.static_params['model_config']['class_mlp_size']
        self.attn_mlp_size = self.static_params['model_config']['attn_mlp_size']
        self.dropout = L.Dropout(1 - self.hparams.mlp_dropout)
        self.dense1 = L.Dense(2 * (self.class_mlp_size + self.attn_mlp_size))
        self.bilinear = nn_utils.BilinearClassifier(1, 1 - self.hparams.bilinear_dropout)

    def make_call(self, data, **kwargs):
        features, mask = data
        # todo AG check all dense for dropout noise shape

        features = self.dense1(features)
        dep_mlp, head_mlp = tf.split(value=features, num_or_size_splits=2, axis=-1)

        dep_arc_mlp, dep_rel_mlp = dep_mlp[:, :, :self.attn_mlp_size], dep_mlp[:, :, self.attn_mlp_size:]
        head_arc_mlp, head_rel_mlp = head_mlp[:, :, :self.attn_mlp_size], head_mlp[:, :, self.attn_mlp_size:]

        # [batch_size x seq_len x seq_len]
        arc_logits = self.bilinear([dep_arc_mlp, head_arc_mlp])
        arc_logits = tf.squeeze(arc_logits, axis=2)

        predictions = tf.argmax(arc_logits, -1)
        probabilities = tf.nn.softmax(arc_logits)

        output = {
          'predictions': predictions,  # [BATCH_SIZE, SEQ_LEN] (predictions for arcs)
          'probabilities': probabilities,  # [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
          'scores': arc_logits,  # [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
          'dep_rel_mlp': dep_rel_mlp,  # [BATCH_SIZE, SEQ_LEN, class_mlp_size]
          'head_rel_mlp': head_rel_mlp  # [BATCH_SIZE, SEQ_LEN, class_mlp_size]
        }

        return output

    def loss(self, targets, output, mask):
        logits = output['scores']
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        n_tokens = tf.reduce_sum(mask)
        return tf.reduce_sum(cross_entropy * mask) / n_tokens


class ConditionalBilinear(OutputLayer):
    def __init__(self, transformer_layer_id, **params):
        super(ConditionalBilinear, self).__init__(transformer_layer_id, **params)
        self.cond_bilinear = nn_utils.ConditionalBilinearClassifier(
            self.static_params['task_vocab_size'],
            1 - self.hparams.bilinear_dropout
        )

    def make_call(self, data, dep_rel_mlp, head_rel_mlp, parse_preds_train, parse_preds_eval, **kwargs):
        # todo AG check if we unpack agruments or ignore data
        parse_preds = parse_preds_train if not self.in_eval_mode else parse_preds_eval
        logits, _ = self.cond_bilinear([dep_rel_mlp, head_rel_mlp, parse_preds])
        predictions = tf.argmax(logits, -1)
        probabilities = tf.nn.softmax(logits)

        output = {
            'scores': logits,  # [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
            'predictions': predictions,  # [BATCH_SIZE, SEQ_LEN] (conditional arcs)
            'probabilities': probabilities,  # [BATCH_SIZE, SEQ_LEN, SEQ_LEN]
        }

        return output

    def loss(self, targets, output, mask):
        logits = output['scores']
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
        n_tokens = tf.reduce_sum(mask)
        return tf.reduce_sum(cross_entropy * mask) / n_tokens


class SRLBilinear(OutputLayer):
    def __init__(self, transformer_layer_id, **params):
        super(SRLBilinear, self).__init__(transformer_layer_id, **params)

        self.predicate_mlp_size = self.static_params['model_config']['predicate_mlp_size']
        self.role_mlp_size = self.static_params['model_config']['role_mlp_size']

        self.dropout = L.Dropout(1 - self.hparams.mlp_dropout)
        self.dense1 = L.Dense(self.predicate_mlp_size + self.role_mlp_size)

        self.bilinear = nn_utils.BilinearClassifier(
            self.static_params['task_vocab_size'],
            1 - self.hparams.bilinear_dropout
        )

    def slice(self):
        pass

    @staticmethod
    def bool_mask_where_predicates(predicates_tensor, mask):
        # TODO this should really be passed in, not assumed...
        predicate_outside_idx = 0
        return tf.logical_and(tf.not_equal(predicates_tensor, predicate_outside_idx), tf.cast(mask, tf.bool))

    def make_call(self, data, predicate_preds_train, predicate_preds_eval, predicate_targets):
        '''

        :param features: [BATCH_SIZE, SEQ_LEN, hidden_size]
        :param mask: [BATCH_SIZE, SEQ_LEN]
        :param predicate_preds: [BATCH_SIZE, SEQ_LEN] Predictions from predicates layer with dims
        :param targets: [BATCH_SIZE, SEQ_LEN, batch_num_predicates] SRL labels
        :param transition_params: [num_labels x num_labels] transition parameters, if doing Viterbi decoding
        '''
        features, mask = data

        input_shape = tf.shape(features)
        batch_size = input_shape[0]
        batch_seq_len = input_shape[1]

        # indices of predicates
        predicate_preds = predicate_preds_train if not self.in_eval_mode else predicate_preds_eval
        # [PRED_COUNT, 2] (batch_row, sentence_pos for each predicate)
        predicate_gather_indices = tf.where(self.bool_mask_where_predicates(predicate_preds, mask))

        # (1) project into predicate, role representations
        features = self.dropout(features)
        predicate_role_mlp = self.dense1(features)  # [BATCH_SIZE, SEQ_LEN, predicate_mlp_size+role_mlp_size]
        predicate_mlp = predicate_role_mlp[:, :, :self.predicate_mlp_size]  # [BATCH_SIZE, SEQ_LEN, predicate_mlp_size]
        role_mlp = predicate_role_mlp[:, :, self.predicate_mlp_size:]  # [BATCH_SIZE, SEQ_LEN, role_mlp_size]

        # (2) feed through bilinear to obtain scores
        # gather just the predicates
        # gathered_predicates: num_predicates_in_batch x 1 x predicate_mlp_size
        # role mlp: batch x seq_len x role_mlp_size
        # gathered roles: need a (batch_seq_len x role_mlp_size) role representation for each predicate,
        # i.e. a (num_predicates_in_batch x batch_seq_len x role_mlp_size) tensor
        gathered_predicates = tf.expand_dims(tf.gather_nd(predicate_mlp, predicate_gather_indices), 1)  # [PRED_COUNT, 1, role_mlp_size]

        # AG duplicate dimension
        tiled_roles = tf.reshape(tf.tile(role_mlp, [1, batch_seq_len, 1]),
                                 [batch_size, batch_seq_len, batch_seq_len, self.role_mlp_size])
        gathered_roles = tf.gather_nd(tiled_roles, predicate_gather_indices)  # [PRED_COUNT, SEQ_LEN, HID]

        # now multiply them together to get (num_predicates_in_batch x batch_seq_len x num_srl_classes) tensor of scores
        srl_logits = self.bilinear([gathered_predicates, gathered_roles])  # [PRED_COUNT, bilin_output_size, SEQ_LEN]
        logits_shape = srl_logits.get_shape()
        srl_logits = tf.reshape(srl_logits, [-1, logits_shape[2], logits_shape[3]])
        srl_logits_transposed = tf.transpose(srl_logits, [0, 2, 1])  # [PRED_COUNT, SEQ_LEN, bilin_output_size]

        # num_predicates_in_batch x seq_len
        predictions = tf.cast(tf.argmax(srl_logits_transposed, axis=-1), tf.int32)  # [PRED_COUNT, SEQ_LEN]

        # todo AG check
        # seq_lens = tf.cast(tf.reduce_sum(gather_mask, 1), tf.int32)
        seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)  # [BATCH_SIZE]

        transition_params = self.static_params["transition_params"]
        if transition_params is not None and self.in_eval_mode:
            predictions, _ = crf_decode(srl_logits_transposed, transition_params, seq_lens)

        # todo AG clear the mess
        output = {
            'predictions': predictions,
            'scores': srl_logits_transposed,  # [PRED_COUNT, SEQ_LEN, bilin_output_size]
            # 'targets': srl_targets_gold_predicates,
            'probabilities': tf.nn.softmax(srl_logits_transposed, -1),  # [PRED_COUNT, SEQ_LEN, bilin_output_size]
            'params': {
                'predicate_preds': predicate_preds,
                'batch_seq_len': batch_seq_len,
                'batch_size': batch_size,
                'predicate_targets': predicate_targets,
            }
        }

        return output

    def loss(self, targets, output, mask):
        srl_logits_transposed = output['scores']
        num_labels = self.static_params['task_vocab_size']
        predicate_preds = output['params']['predicate_preds']
        batch_seq_len = output['params']['batch_seq_len']
        batch_size = output['params']['batch_size']
        predicate_targets = output['params']['predicate_targets']
        seq_lens = tf.cast(tf.reduce_sum(mask, 1), tf.int32)
        transition_params = self.static_params['transition_params']

        predicate_gather_indices = tf.where(self.bool_mask_where_predicates(predicate_preds, mask))

        # (3) compute loss
        # need to repeat each of these once for each target in the sentence
        mask_tiled = tf.reshape(tf.tile(mask, [1, batch_seq_len]), [batch_size, batch_seq_len, batch_seq_len])
        gather_mask = tf.gather_nd(mask_tiled, predicate_gather_indices)

        # now we have k sets of targets for the k frames
        # (p1) f1 f2 f3
        # (p2) f1 f2 f3

        # get all the tags for each token (which is the predicate for a frame), structuring
        # targets as follows (assuming p1 and p2 are predicates for f1 and f3, respectively):
        # (p1) f1 f1 f1
        # (p2) f3 f3 f3
        srl_targets_transposed = tf.transpose(targets, [0, 2, 1])

        gold_predicate_counts = tf.reduce_sum(
            tf.cast(self.bool_mask_where_predicates(predicate_targets, mask), tf.int32), -1)
        srl_targets_indices = tf.where(tf.sequence_mask(tf.reshape(gold_predicate_counts, [-1])))

        # num_predicates_in_batch x seq_len
        srl_targets_gold_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_indices)

        predicted_predicate_counts = tf.reduce_sum(tf.cast(
            self.bool_mask_where_predicates(predicate_preds, mask), tf.int32), -1)
        srl_targets_pred_indices = tf.where(tf.sequence_mask(tf.reshape(predicted_predicate_counts, [-1])))
        srl_targets_predicted_predicates = tf.gather_nd(srl_targets_transposed, srl_targets_pred_indices)

        if transition_params is not None and not self.in_eval_mode:
            log_likelihood, transition_params = crf_log_likelihood(
                srl_logits_transposed,
                srl_targets_predicted_predicates,
                seq_lens, transition_params
            )
            loss = tf.reduce_mean(-log_likelihood)
        else:
            srl_targets_onehot = tf.one_hot(indices=srl_targets_predicted_predicates, depth=num_labels, axis=-1)
            loss = tf.compat.v1.losses.softmax_cross_entropy(
                logits=tf.reshape(srl_logits_transposed, [-1, num_labels]),
                onehot_labels=tf.reshape(srl_targets_onehot, [-1, num_labels]),
                weights=tf.reshape(gather_mask, [-1]),
                label_smoothing=self.hparams.label_smoothing,
                reduction=tf.compat.v1.losses.Reduction.SUM_BY_NONZERO_WEIGHTS,
            )

        return loss


dispatcher = {
  'srl_bilinear': SRLBilinear,
  'joint_softmax_classifier': JointSoftmaxClassifier,
  'softmax_classifier': SoftmaxClassifier,
  'parse_bilinear': ParseBilinear,
  'conditional_bilinear': ConditionalBilinear,
}
