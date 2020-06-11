import tensorflow as tf
import numpy as np
from sklearn.metrics import f1_score
from collections import defaultdict

import evaluation_fns_np
import nn_utils
import util

EPS = 1e-20


class BaseMetric:
    def __init__(self, task, config, reverse_maps=None):
        self.task = task
        self.static_params = {}
        self.label_params = {}
        self.token_params = {}
        self.output_params = {}

        for param_name, param_values in config.get('params', {}).items():
            if 'label' in param_values:
                self.label_params[param_name] = param_values['label']
            elif 'feature' in param_values:
                self.token_params[param_name] = param_values['feature']
            elif 'layer' in param_values:
                self.output_params[param_name] = param_values['layer'], param_values['output']
            elif 'reverse_maps' in param_values:
                self.static_params[param_name] = {
                    map_name: reverse_maps[map_name] for map_name in param_values['reverse_maps']
                }
            else:
                self.static_params[param_name] = param_values['value']

        self.history = []

    def reset_states(self):
        self.history = []

    def update_state(self, values):
        self.history.append(values)

    def result(self):
        raise NotImplementedError

    def update(self, labels, predictions):
        task_outputs = predictions[self.task]['predictions']
        task_labels = labels[self.task]
        tokens = predictions['tokens']
        mask = predictions['mask']

        labels = {key: labels[value] for key, value in self.label_params.items()}
        outputs = {key: predictions[layer_name].get(field_name)  # todo remove extra args from configs
                   for key, (layer_name, field_name) in self.output_params.items()}
        tokens = {key: tokens[value] for key, value in self.token_params.items()}

        params = {}
        params.update(labels)
        params.update(outputs)
        params.update(tokens)
        return self.make_call(task_labels, task_outputs, mask, **params)

    def make_call(self, labels, outputs, mask, **kwargs):
        raise NotImplementedError


class Accuracy(BaseMetric):
    name = "Accuracy"

    def __init__(self, *args, **kwargs):
        super(Accuracy, self).__init__(*args, **kwargs)

    def reset_states(self):
        self.accuracy = tf.keras.metrics.Accuracy()

    def make_call(self, labels, outputs, mask, **kwargs):
        """
        :param labels: [BATCH_SIZE, SEQ_LEN]
        :param outputs: [BATCH_SIZE, SEQ_LEN]
        :param mask: [BATCH_SIZE, SEQ_LEN]
        """

        self.accuracy.update_state(labels, outputs, sample_weight=mask)

    def result(self):
        return self.accuracy.result().numpy()


class ConllSrlEval(BaseMetric):
    """
    A proxy between the model and conll scripts for metric computation
    """
    name = "ConllSrlEval"

    def make_call(self, labels, outputs, mask, words,
                  predicate_predictions, predicate_targets, pos_predictions, pos_targets, **kwargs):

        reverse_maps = self.static_params['reverse_maps']
        gold_srl_eval_file = self.static_params['gold_srl_eval_file']
        pred_srl_eval_file = self.static_params['pred_srl_eval_file']

        # create accumulator variables
        correct_count = 0
        excess_count = 0
        missed_count = 0

        # first, use reverse maps to convert ints to strings
        str_predictions = nn_utils.int_to_str_lookup_table(outputs, reverse_maps['srl'])
        str_words = nn_utils.int_to_str_lookup_table(words, reverse_maps['word'])
        str_targets = nn_utils.int_to_str_lookup_table(labels, reverse_maps['srl'])
        str_pos_predictions = nn_utils.int_to_str_lookup_table(pos_predictions, reverse_maps['gold_pos'])
        str_pos_targets = nn_utils.int_to_str_lookup_table(pos_targets, reverse_maps['gold_pos'])

        if 'srl_mask' in kwargs:
            srl_mask = kwargs['srl_mask']
            srl_seq_len = tf.math.reduce_max(tf.reduce_sum(srl_mask, -1))
            # no need for extra padding, as WORD_SEQ_LEN is a length of a padded word sequence
            srl_mask = util.padded_to_full_word_mask(srl_mask, srl_seq_len, tf.shape(srl_mask)[1])

            str_predictions = util.take_word_start_tokens(str_predictions, srl_mask)
            predicate_predictions = util.take_word_start_tokens(predicate_predictions, srl_mask)
            str_words = util.take_word_start_tokens(str_words, srl_mask)
            mask = util.take_word_start_tokens(mask, srl_mask)
            str_targets = util.take_word_start_tokens(str_targets, srl_mask)
            predicate_targets = util.take_word_start_tokens(predicate_targets, srl_mask)
            str_pos_predictions = util.take_word_start_tokens(str_pos_predictions, srl_mask)
            str_pos_targets = util.take_word_start_tokens(str_pos_targets, srl_mask)

        # need to pass through the stuff for pyfunc
        # pyfunc is necessary here since we need to write to disk
        py_eval_inputs = [str_predictions, predicate_predictions, str_words, mask, str_targets, predicate_targets,
                          pred_srl_eval_file, gold_srl_eval_file, str_pos_predictions, str_pos_targets]
        # out_types = [tf.int64, tf.int64, tf.int64]
        # correct, excess, missed = tf.py_function(evaluation_fns_np.conll_srl_eval, py_eval_inputs, out_types)
        correct, excess, missed = evaluation_fns_np.conll_srl_eval(*[
            x.numpy() if isinstance(x, tf.Tensor) else x for x in py_eval_inputs])

        correct_count += correct
        excess_count += excess
        missed_count += missed

        self.update_state([correct_count, excess_count, missed_count])

    def result(self):
        data = np.array(self.history)
        correct_count = data[:, 0].sum()
        excess_count = data[:, 1].sum()
        missed_count = data[:, 2].sum()

        precision = correct_count / (correct_count + excess_count + EPS)
        recall = correct_count / (correct_count + missed_count + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)

        return precision, recall, f1


class ConllParseEval(BaseMetric):
    """
    A proxy between the model and conll scripts for metric computation
    """
    name = "ConllParseEval"

    def make_call(self, labels, outputs, mask, words,
                  parse_head_predictions, parse_head_targets, pos_targets, **kwargs):
        reverse_maps = self.static_params['reverse_maps']
        gold_parse_eval_file = self.static_params['gold_parse_eval_file']
        pred_parse_eval_file = self.static_params['pred_parse_eval_file']

        total_count = 0
        correct_count = np.zeros(3)

        str_words = nn_utils.int_to_str_lookup_table(words, reverse_maps['word'])
        str_predictions = nn_utils.int_to_str_lookup_table(outputs, reverse_maps['parse_label'])
        str_targets = nn_utils.int_to_str_lookup_table(labels, reverse_maps['parse_label'])
        str_pos_targets = nn_utils.int_to_str_lookup_table(pos_targets, reverse_maps['gold_pos'])

        # need to pass through the stuff for pyfunc
        # pyfunc is necessary here since we need to write to disk
        py_eval_inputs = [str_predictions, parse_head_predictions, str_words, mask, str_targets, parse_head_targets,
                          pred_parse_eval_file, gold_parse_eval_file, str_pos_targets]

        # out_types = [tf.int64, tf.int64]
        # total, corrects = tf.py_function(evaluation_fns_np.conll_parse_eval, py_eval_inputs, out_types)
        total, corrects = evaluation_fns_np.conll_parse_eval(*[
            x.numpy() if isinstance(x, tf.Tensor) else x for x in py_eval_inputs])
        self.update_state([total, corrects])

    def result(self):
        data = np.array(self.history)
        total_count = data[:, 0].sum()
        correct_count = data[:, 1].sum()
        return correct_count / total_count


class LabelF1(BaseMetric):
    name = "LabelF1"

    def make_call(self, labels, outputs, mask, **kwargs):
        reverse_maps = self.static_params['reverse_maps']
        outputs = nn_utils.int_to_str_lookup_table(outputs, reverse_maps['srl'])
        labels = nn_utils.int_to_str_lookup_table(labels, reverse_maps['srl'])

        if 'srl_mask' in kwargs:
            srl_mask = kwargs['srl_mask']
            srl_seq_len = tf.math.reduce_max(tf.reduce_sum(srl_mask, -1))
            # no need for extra padding, as WORD_SEQ_LEN is a length of a padded word sequence
            srl_mask = util.padded_to_full_word_mask(srl_mask, srl_seq_len, tf.shape(srl_mask)[1])

            labels = util.take_word_start_tokens(labels, srl_mask)
            outputs = util.take_word_start_tokens(outputs, srl_mask)
            mask = util.take_word_start_tokens(mask, srl_mask)

        labels = labels.numpy().reshape(-1)
        outputs = outputs.numpy().reshape(-1)
        mask = mask.numpy().reshape(-1)

        l_list, o_list = [], []
        for l, o, m in zip(labels, outputs, mask):
            if l != "*" and o != "*" and m:
                l_list.append(l)
                o_list.append(o)

        self.update_state([l_list, o_list])

    def result(self):
        labels = []
        outputs = []
        for l, o in self.history:
            labels.extend(l)
            outputs.extend(o)

        f1_macro = f1_score(labels, outputs, average='macro')
        f1_micro = f1_score(labels, outputs, average='micro')
        return [f1_macro, f1_micro]


class ArgumentDetectionF1(BaseMetric):
    """
    F1 on argument detection, regardless of classification
    """
    name = "ArgumentDetectionF1"

    def __init__(self, *args, **kwargs):
        super(ArgumentDetectionF1, self).__init__(*args, **kwargs)
        self.conll_srl = ConllSrlEval(*args, **kwargs)

        forward_srl_map = {v: k for k, v in self.static_params['reverse_maps']['srl']}
        out_label = forward_srl_map["O"]
        v_label = forward_srl_map["B-V"]
        in_label = forward_srl_map["B-ARG0"] if "B-ARG0" in forward_srl_map else forward_srl_map["B-агенс"]  # todo to config
        self.mapper = lambda x: x if x in [out_label, v_label] else in_label

    def make_call(self, labels, outputs, *args, **kwargs):
        labels = tf.map_fn(self.mapper, labels)
        outputs = tf.map_fn(self.mapper, outputs)
        self.conll_srl.make_call(labels, outputs, *args, **kwargs)

    def reset_states(self):
        self.conll_srl.reset_states()

    def result(self):
        return self.conll_srl.result()


class ValLoss(BaseMetric):
    name = "ValLosses"

    def __init__(self):
        super(ValLoss, self).__init__(None, {})

    def update(self, labels, outputs, **kwargs):
        self.update_state(outputs)

    def result(self):
        # this might fail when run in the graph
        # not fixing it yet because eval_fns_np needs rewriting and srl_output has a bug for graphs
        return tf.reduce_mean(self.history, 0).numpy()


dispatcher = {
    'accuracy': Accuracy,
    'conll_srl_eval': ConllSrlEval,
    'conll_parse_eval': ConllParseEval,
    'label_f1': LabelF1,
    'arg_detection_f1': ArgumentDetectionF1,
}
