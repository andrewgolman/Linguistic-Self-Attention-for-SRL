import transformers
import tensorflow as tf
try:
    import torch
except ImportError:
    print("UNABLE TO IMPORT TORCH")

from opennmt.utils.misc import shape_list


class RubertRunner(tf.keras.models.Model):
    """
    Runs RuBert from deeppavlov, using pytorch model from transformers library

    Don't beat me up, as far as I know, this is the simplest way to run rubert on keras.
    Well, not actually on keras. But it works.
    bert-for-tf2 library looks promising, though
    """
    def __init__(self):
        super(RubertRunner, self).__init__()
        self.model = transformers.BertModel.from_pretrained("pretrained/rubert_cased_L-12_H-768_A-12_pt")

    def call(self, x):
        shape = shape_list(x)
        shape.append(768)
        x = tf.numpy_function(self.call_model, [x], tf.float32)
        return tf.reshape(x, tf.stack(shape))

    def call_model(self, x):
        with torch.no_grad():
            x = torch.tensor(x, dtype=torch.long)
            x = self.model(x)[0]
            return x.numpy()


albert_tokenizer = transformers.AlbertTokenizer.from_pretrained('albert-base-v1')
t5_tokenizer = transformers.T5Tokenizer.from_pretrained('t5-base')
bert_tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-cased')

vocab_file = "pretrained/rubert_cased_L-12_H-768_A-12_v2/vocab.txt"

models = {
    'albert': (transformers.TFAlbertModel, 'albert-base-v1'),
    't5': (transformers.TFT5Model, 't5-base'),
    'bert': (transformers.TFBertModel, 'bert-base-cased'),
    'rubert': (RubertRunner, lambda x: x()),
}


def load_model(name):
    model_class, loader = models[name]
    if isinstance(loader, str):
        return model_class.from_pretrained(loader)
    else:
        return loader(model_class)
