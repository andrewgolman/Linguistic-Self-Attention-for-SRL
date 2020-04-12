import tensorflow as tf
import evaluation_fns_np
import nn_utils
import numpy as np

EPS = 1e-20


class EvalFunction:
    def __init__(self, task, config, reverse_maps=None):
        self.task = task
        self.static_params = {}
        self.label_params = {}
        self.token_params = {}
        self.output_params = {}
        self.requires_transformation = 'params' in config

        if self.requires_transformation:
            for param_name, param_values in config['params'].items():
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

    def __call__(self, labels, predictions):
        task_outputs = predictions[self.task]['predictions']
        task_labels = labels[self.task]
        tokens = predictions['tokens']
        mask = predictions['mask']

        labels = {key: labels.get(value) for key, value in self.label_params.items()}
        outputs = {key: predictions.get(layer_name, {}).get(field_name)
                   for key, (layer_name, field_name) in self.output_params.items()}
        tokens = {key: tokens.get(value) for key, value in self.token_params.items()}

        params = {}
        params.update(labels)
        params.update(outputs)
        params.update(tokens)
        return self.make_call(task_labels, task_outputs, mask, **params)

    def make_call(self, labels, outputs, mask, **kwargs):
        raise NotImplementedError


class Accuracy(EvalFunction):
    def make_call(self, labels, outputs, mask, **kwargs):
        """
        :param labels: [BATCH_SIZE, SEQ_LEN]
        :param outputs: [BATCH_SIZE, SEQ_LEN]
        :param mask: [BATCH_SIZE, SEQ_LEN]
        :return: todo AG ops or number?
        """
        instance = tf.keras.metrics.Accuracy()
        instance.update_state(labels, outputs, sample_weight=mask)
        return instance.result().numpy()


class ConllSrlEval(EvalFunction):
    def make_call(self, labels, outputs, mask, words, **kwargs):

        reverse_maps = self.static_params['reverse_maps']
        # todo AG wtf
        predicate_predictions = kwargs['predicate_predictions']
        predicate_targets = kwargs['predicate_targets']
        gold_srl_eval_file = self.static_params['gold_srl_eval_file']
        pred_srl_eval_file = self.static_params['pred_srl_eval_file']
        pos_predictions = kwargs['pos_predictions']
        pos_targets = kwargs['pos_targets']

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


        # need to pass through the stuff for pyfunc
        # pyfunc is necessary here since we need to write to disk
        py_eval_inputs = [str_predictions, predicate_predictions, str_words, mask, str_targets, predicate_targets,
                          pred_srl_eval_file, gold_srl_eval_file, str_pos_predictions, str_pos_targets]
        out_types = [tf.int64, tf.int64, tf.int64]
        correct, excess, missed = evaluation_fns_np.conll_srl_eval(*[
            x.numpy() if isinstance(x, tf.Tensor) else x for x in py_eval_inputs])
        # correct, excess, missed = tf.py_function(evaluation_fns_np.conll_srl_eval, py_eval_inputs, out_types)

        correct_count += correct
        excess_count += excess
        missed_count += missed

        precision = correct_count / (correct_count + excess_count + EPS)
        recall = correct_count / (correct_count + missed_count + EPS)
        f1 = 2 * precision * recall / (precision + recall + EPS)

        return f1


class ConllParseEval(EvalFunction):
    def make_call(self, labels, outputs, mask, words, **kwargs):
        reverse_maps = self.static_params['reverse_maps']
        # todo AG wtf
        parse_head_predictions = kwargs['parse_head_predictions']
        parse_head_targets = kwargs['parse_head_targets']
        gold_parse_eval_file = self.static_params['gold_parse_eval_file']
        pred_parse_eval_file = self.static_params['pred_parse_eval_file']
        pos_targets = kwargs['pos_targets']

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
        out_types = [tf.int64, tf.int64]
        total, corrects = evaluation_fns_np.conll_parse_eval(*py_eval_inputs)
        # total, corrects = tf.py_func(evaluation_fns_np.conll_parse_eval, py_eval_inputs, out_types, stateful=False)

        total_count += total
        correct_count += corrects
        accuracies = correct_count / total_count

        return accuracies


dispatcher = {
  'accuracy': Accuracy,
  'conll_srl_eval': ConllSrlEval,
  'conll_parse_eval': ConllParseEval,
}
