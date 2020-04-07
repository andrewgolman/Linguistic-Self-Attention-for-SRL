import tensorflow.keras as keras


class FunctionDispatcher(keras.layers.Layer):
    def __init__(self, config, **kwargs):
        super(FunctionDispatcher, self).__init__()
        self.in_eval_mode = False

        self.static_params = kwargs
        self.label_params = {}
        self.token_params = {}
        self.output_params = {}
        self.embedding_params = {}

        # self.vocab = vocab

        for param_name, param_values in config['params'].items():
            # if this is a map-type param, do map lookups and pass those through
            if 'label' in param_values:
                self.label_params[param_name] = param_values['label']
            elif 'feature' in param_values:
                self.token_params[param_name] = param_values['feature']
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
                # else: raise KeyError("Config requires vocab passed to FunctionDispatcher")
            else:
                self.static_params[param_name] = param_values['value']

    def set_train_mode(self):
        self.in_eval_mode = True

    def set_eval_mode(self):
        self.in_eval_mode = False

    def call(self, features=None, outputs=None, tokens=None, labels=None, embeddings=None):
        labels = {key: labels.get(value) for key, value in self.label_params.items()} if labels else {}
        outputs = {key: outputs.get(layer_name, {}).get(field_name) for key, (layer_name, field_name) in self.output_params.items()} if outputs else {}
        tokens = {key: tokens.get(value) for key, value in self.token_params.items()} if tokens else {}
        embeddings = {key: embeddings.get(value) for key, value in self.embedding_params.items()} if embeddings else {}
        return self.make_call(features, **labels, **tokens, **outputs, **embeddings)

    def make_call(self, features, **kwargs):
        raise NotImplementedError
