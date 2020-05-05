import transformers

albert_tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v1')
t5_tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

models = {
    'albert': (transformers.TFAlbertModel, 'albert-base-v1'),
    't5': (transformers.TFT5Model, 't5-base'),
    'bert': (transformers.TFBertModel, 'bert-base-cased'),
}


def load_model(name):
    model_class, weights = models[name]
    return model_class.from_pretrained(weights)