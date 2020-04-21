import tensorflow as tf
import numpy as np

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
        :return: todo AG ops or number?
        """

        self.accuracy.update_state(labels, outputs, sample_weight=mask)

    def result(self):
        return self.accuracy.result().numpy()


class ConllSrlEval(BaseMetric):
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


        # need to pass through the stuff for pyfunc
        # pyfunc is necessary here since we need to write to disk
        py_eval_inputs = [str_predictions, predicate_predictions, str_words, mask, str_targets, predicate_targets,
                          pred_srl_eval_file, gold_srl_eval_file, str_pos_predictions, str_pos_targets]
        out_types = [tf.int64, tf.int64, tf.int64]
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

        return f1


class ConllParseEval(BaseMetric):
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
        total, corrects = evaluation_fns_np.conll_parse_eval(*py_eval_inputs)
        self.update_state([total, corrects])

    def result(self):
        data = np.array(self.history)
        total_count = data[:, 0].sum()
        correct_count = data[:, 1].sum()
        return correct_count / total_count


dispatcher = {
  'accuracy': Accuracy,
  'conll_srl_eval': ConllSrlEval,
  'conll_parse_eval': ConllParseEval,
}


class CallMetricsCallback(tf.keras.callbacks.Callback):
    """
    todo docs
    """
    def __init__(self, metrics, validation_data):
        super(CallMetricsCallback, self).__init__()
        self.metrics = metrics
        self.X_val, self.Y_val = validation_data

    def run_eval(self):
        Y_pred = self.model.predict(self.X_val)
        scores = {}
        for metric in self.metrics:
            scores[(metric.task, metric.name)] = metric(self.Y_val, Y_pred)
        return scores

    def on_epoch_end(self, epoch, logs={}):
        scores = self.run_eval()
        print("EPOCH:", epoch)
        for k, v in scores.items():
            print(k, ":", v)


def print_model_metrics(model):
    # losses = model.get_losses()
    metrics = model.get_metrics()
    # print("Validation losses:")
    # for i, v in enumerate(losses):
    #     print(i, ":", v)
    print("Validation metrics:")
    for k, v in metrics.items():
        print(k, ":", v)


class EvalMetricsCallBack(tf.keras.callbacks.Callback):
    """
    todo docs
    """
    def __init__(self, dataset):
        super(EvalMetricsCallBack, self).__init__()
        self.ds = dataset

    def on_epoch_end(self, epoch, logs={}):
        self.model.start_custom_eval()
        for batch in self.ds.as_numpy_iterator():
            self.model(batch)
            # self.model.predict(batch)

        print("=" * 20)
        print("EPOCH:", epoch + 1)
        print_model_metrics(self.model)
        self.model.end_custom_eval()
