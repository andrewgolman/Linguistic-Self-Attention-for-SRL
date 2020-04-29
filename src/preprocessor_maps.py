import transformers

albert_tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v1')

models = {
    'albert': (transformers.TFAlbertModel, 'albert-base-v1')
}


def load_model(name):
    model_class, weights = models[name]
    return model_class.from_pretrained(weights)
