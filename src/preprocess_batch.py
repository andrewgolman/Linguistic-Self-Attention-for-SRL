import tensorflow as tf
import constants
import util
import tensorflow.compat.v1.logging as logging


# todo AG check that embeddings can be trained
class LISAModelPreprocess:
    def __init__(self, feature_idx_map, label_idx_map, model_config, task_list, vocab):
        self.feature_idx_map = feature_idx_map
        self.label_idx_map = label_idx_map
        self.model_config = model_config
        self.task_list = task_list
        self.vocab = vocab
        self.embeddings = self.get_embeddings()

    def get_embedding_table(self, name, embedding_dim, include_oov, pretrained_fname=None, num_embeddings=None):
        """
        AG: get table, check dimension, add oov and convert to tensor
        """
        if pretrained_fname:
            pretrained_embeddings = util.load_pretrained_embeddings(pretrained_fname)
            pretrained_num_embeddings, pretrained_embedding_dim = pretrained_embeddings.shape
            if pretrained_embedding_dim != embedding_dim:
                util.fatal_error("Pre-trained %s embedding dim does not match specified dim (%d vs %d)." %
                                 (name, pretrained_embedding_dim, embedding_dim))
            if num_embeddings and num_embeddings != pretrained_num_embeddings:
                util.fatal_error("Number of pre-trained %s embeddings does not match specified "
                                 "number of embeddings (%d vs %d)." % (name, pretrained_num_embeddings, num_embeddings))
            embedding_table = tf.convert_to_tensor(pretrained_embeddings, dtype=tf.float32)
        else:
           embedding_table = tf.random.normal(shape=[num_embeddings, embedding_dim])

        if include_oov:
            oov_embedding = tf.random.normal(shape=[1, embedding_dim])
            embedding_table = tf.concat([embedding_table, oov_embedding], axis=0)

        return embedding_table

    def get_embeddings(self):
        # Create embeddings tables, loading pre-trained if specified
        embeddings = {}
        for embedding_name, embedding_map in self.model_config['embeddings'].items():
            embedding_dim = embedding_map['embedding_dim']
            if 'pretrained_embeddings' in embedding_map:
                input_pretrained_embeddings = embedding_map['pretrained_embeddings']
                include_oov = True
                embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                         pretrained_fname=input_pretrained_embeddings)
            else:
                num_embeddings = self.vocab.vocab_names_sizes[embedding_name]
                include_oov = self.vocab.oovs[embedding_name]
                embedding_table = self.get_embedding_table(embedding_name, embedding_dim, include_oov,
                                                       num_embeddings=num_embeddings)
            embeddings[embedding_name] = embedding_table
            logging.log(logging.INFO, "Created embeddings for '%s'." % embedding_name)
        return embeddings

    def __call__(self, batch):
        batch_shape = tf.shape(batch)
        batch_size = batch_shape[0]
        batch_seq_len = batch_shape[1]

        features = {f: batch[:, :, idx] for f, idx in self.feature_idx_map.items()}

        # ES todo this assumes that word_type is always passed in
        words = features['word_type']

        # for masking out padding tokens
        mask = tf.where(tf.equal(words, constants.PAD_VALUE), tf.zeros([batch_size, batch_seq_len]),
                        tf.ones([batch_size, batch_seq_len]))
        # Extract named features from monolithic "features" input
        features = {f: tf.multiply(tf.cast(mask, tf.int32), v) for f, v in features.items()}
        tokens = features.copy()

        # Extract named labels from monolithic "features" input, and mask them
        # ES todo fix masking -- is it even necessary?
        labels = {}
        for l, idx in self.label_idx_map.items():
            these_labels = batch[:, :, idx[0]:idx[1]] if idx[1] != -1 else batch[:, :, idx[0]:]
            these_labels_masked = tf.multiply(these_labels, tf.cast(tf.expand_dims(mask, -1), tf.int32))
            # check if we need to mask another dimension
            if idx[1] == -1:
                last_dim = tf.shape(these_labels)[2]
                this_mask = tf.where(tf.equal(these_labels_masked, constants.PAD_VALUE),
                                     tf.zeros([batch_size, batch_seq_len, last_dim], dtype=tf.int32),
                                     tf.ones([batch_size, batch_seq_len, last_dim], dtype=tf.int32))
                these_labels_masked = tf.multiply(these_labels_masked, this_mask)
            else:
                these_labels_masked = tf.squeeze(these_labels_masked, -1)
            labels[l] = these_labels_masked

        # Set up model inputs
        features = [
            tf.nn.embedding_lookup(self.embeddings[input_name], features[input_name])
            for input_name in self.model_config['inputs']  # word type and/or predicate
        ]
        features = tf.concat(features, axis=2)

        labels = util.dict2list(labels, self.label_idx_map.keys())
        tokens = util.dict2list(tokens, self.feature_idx_map.keys())

        return [features, mask, *labels, *tokens], [0, 0, 0, 0]  # todo AG np.zeros(len(task_list) + 1)
