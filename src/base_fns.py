import tensorflow.keras as keras


class FunctionDispatcher(keras.layers.Layer):
    """
    Proxy between a model and a function-layer.
    Finds relevant arguments according to configs (see config/readme.md) and passes it to the function-layer.
    """
    def __init__(self, config, **kwargs):
        super(FunctionDispatcher, self).__init__()
        self.in_eval_mode = False  # used only for output layers
        self.teacher_forcing = False

        self.static_params = kwargs
        self.label_params = {}
        self.output_params = {}
        self.embedding_params = {}

        for param_name, param_values in config['params'].items():
            # if this is a map-type param, do map lookups and pass those through
            if 'label' in param_values:
                self.label_params[param_name] = param_values['label']
            # otherwise, this is a previous-prediction-type param, look those up and pass through
            elif 'layer' in param_values:
                self.output_params[param_name] = param_values['layer'], param_values['output']
            elif 'embeddings' in param_values:
                self.embedding_params[param_name] = param_values['embeddings']
            # if this is a map-type param, do map lookups and pass those through
            elif 'joint_maps' in param_values:
                joint_lookup_maps = kwargs['joint_lookup_maps']
                self.static_params[param_name] = {
                    map_name: joint_lookup_maps[map_name] for map_name in param_values['joint_maps']
                }
            else:
                self.static_params[param_name] = param_values['value']

    def set_train_mode(self):
        self.in_eval_mode = True

    def set_eval_mode(self):
        self.in_eval_mode = False

    def enable_teacher_forcing(self):
        self.teacher_forcing = True

    def disable_teacher_forcing(self):
        self.teacher_forcing = False

    def call(self, data=None, outputs=None, labels=None):
        """
        :param data: [features(Tensor), mask(Tensor)]
        :param outputs: Dict
        :param labels: Dict
        :param embeddings: Dict
        :return: function output
        """
        labels = {key: labels[value] for key, value in self.label_params.items()} if labels else {}
        outputs = {key: outputs[layer_name][field_name] for key, (layer_name, field_name) in self.output_params.items()} if outputs else {}
        embeddings = {key: self.static_params['embeddings'][value] for key, value in self.embedding_params.items()}
        return self.make_call(data, **labels, **outputs, **embeddings)

    def make_call(self, data, **kwargs):
        """
        Function-layer call method
        :param data: [features(Tensor), mask(Tensor)]
        :param kwargs: function arguments. Might contain excessive arguments  # todo do not pass it
        """
        raise NotImplementedError
