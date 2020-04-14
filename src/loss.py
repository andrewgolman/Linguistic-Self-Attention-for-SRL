import util
import tensorflow as tf


class LISAModelLoss:
    def __init__(self, output_layers, task_config, task_list):
        self.items_to_log = {}
        self.output_layers = output_layers
        self.task_config = task_config
        self.task_list = task_list

    def __call__(self, labels, predictions, sample_weight=None):
        loss = 0

        # todo pass dicts when keras lets it
        labels = util.list2dict(labels, self.task_list)
        # predictions = util.list2dict(predictions, self.task_list)

        for task_data in self.task_config.values():
            for task, task_map in task_data.items():
                this_output_layer = self.output_layers[task]
                this_task_loss = this_output_layer.loss(labels[task], predictions[task], mask=predictions['mask'])

                # self.items_to_log['{}_loss'.format(task)] = this_task_loss
                loss += this_task_loss * task_map['penalty']

        return loss


class DummyLoss:
    def __call__(self, labels, predictions, sample_weight=None):
        return 0.


class SumLoss:
    def __call__(self, labels, predictions, sample_weight=None):
        return tf.reduce_sum(predictions)
