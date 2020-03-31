import tensorflow.keras as keras


class FunctionDispatcher(keras.models.Model):
    def __init__(self, config, vocab=None):
        self.in_eval_mode = False

        self.static_params = {}
        self.label_params = {}
        self.token_params = {}
        self.output_params = {}
        self.embedding_params = {}

        self.vocab = vocab

        for param_name, param_values in config['params'].items():
            # if this is a map-type param, do map lookups and pass those through
            if 'label' in param_values:
                self.label_params[param_name] = param_values['label']
            elif 'feature' in param_values:
                self.token_params[param_name] = param_values['feature']
            # otherwise, this is a previous-prediction-type param, look those up and pass through
            elif 'layer' in param_values:
                self.output_params[param_name] = param_values['layer']
            elif 'embeddings' in param_values:
                self.embedding_params[param_name] = param_values['embeddings']
            # if this is a map-type param, do map lookups and pass those through
            elif 'joint_maps' in param_values:
                if not self.vocab:
                    raise KeyError("Config requires vocab passed to FunctionDispatcher")
                self.static_params[param_name] = {
                    map_name: self.vocab.joint_lookup_maps[map_name] for map_name in param_values['joint_maps']
                }
            else:
                self.static_params[param_name] = param_values['value']

    def set_train_mode(self):
        self.in_eval_mode = True

    def set_eval_mode(self):
        self.in_eval_mode = False

    def __call__(self, features=None, outputs=None, tokens=None, labels=None, embeddings=None):
        labels = {labels[key]: value for key, value in self.label_params}
        outputs = {outputs[key]: value for key, value in self.output_params}
        tokens = {tokens[key]: value for key, value in self.token_params}
        embeddings = {embeddings[key]: value for key, value in self.embedding_params}
        return self.make_call(features, **labels, **tokens, **outputs, **embeddings)

    def make_call(self, features, **kwargs):
        raise NotImplementedError
