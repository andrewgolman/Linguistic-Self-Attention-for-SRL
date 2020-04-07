
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
                this_task_loss = this_output_layer.loss(labels[task], predictions[task], mask=predictions['mask'])

                # self.items_to_log['{}_loss'.format(task)] = this_task_loss
                loss += this_task_loss * task_map['penalty']

                # do the evaluation
                # for eval_name, eval_map in task_map['eval_fns'].items():
                #     eval_fn_params = evaluation_fns.get_params(task_outputs, eval_map, predictions, feats, labels,
                #                                                task_labels, self.vocab.reverse_maps, mask)
                    # eval_result = evaluation_fns.dispatch(eval_map['name'])(**eval_fn_params)
                    # eval_metric_ops[eval_name] = eval_result

        return loss
