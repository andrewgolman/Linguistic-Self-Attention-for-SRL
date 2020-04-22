import tensorflow as tf
import numpy as np
import os
import util
import constants
import data_converters
import tensorflow.compat.v1.logging as logging


class Vocab:
  '''
  Handles creating and caching vocabulary files and tf vocabulary lookup ops for a given list of data files.
  '''

  def __init__(self, data_config, save_dir, data_filenames=None):
    self.data_config = data_config
    self.save_dir = save_dir

    # self.vocab_sizes = {}
    self.joint_label_lookup_maps = {}
    self.reverse_maps = {}
    self.vocab_maps = {}
    self.vocab_lookups = None
    self.oovs = {}

    # make directory for vocabs
    self.vocabs_dir = "%s/assets.extra" % save_dir
    if not os.path.exists(self.vocabs_dir):
      try:
        os.mkdir(self.vocabs_dir)
      except OSError as e:
        util.fatal_error("Failed to create vocabs directory: %s; %s" % (self.vocabs_dir, e.strerror))
      else:
        logging.log(logging.INFO, "Successfully created vocabs directory: %s" % self.vocabs_dir)
    else:
      logging.log(logging.INFO, "Using vocabs directory: %s" % self.vocabs_dir)

    self.vocab_names_sizes = self.make_vocab_files(self.data_config, self.save_dir, data_filenames)



  '''
  Creates tf.contrib.lookup ops for all the vocabs defined in self.data_config.
  
  Args: 
    word_embedding_file: File containing word embedding vocab, with words in the first space-separated column
    
  Returns:
    Map from vocab names to tf.contrib.lookup ops, map from vocab names to vocab sizes
  '''
  def create_vocab_lookup_ops(self, embedding_files=None):

    # Don't waste GPU memory with these lookup tables; tell tf to put it on CPU
    with tf.device('/cpu:0'):
      vocab_lookup_ops = {}
      for v in self.vocab_names_sizes.keys():
        if v in self.data_config:
          num_oov = 1 if 'oov' in self.data_config[v] and self.data_config[v]['oov'] else 0
          # todo
          # this_lookup = tf.contrib.lookup.index_table_from_file("%s/%s.txt" % (self.vocabs_dir, v),
          #                                                       num_oov_buckets=num_oov,
          #                                                       key_column_index=0)
          # vocab_lookup_ops[v] = this_lookup
          vocab_lookup_ops[v] = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
              "%s/%s.txt" % (self.vocabs_dir, v), tf.string, 0, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER,
            ),
            num_oov_buckets=1,  # =num_oov, todo CHECK
          )

      if embedding_files:
        for embedding_file in embedding_files:
          embeddings_name = embedding_file
          print(embedding_file)
          vocab_lookup_ops[embeddings_name] = tf.lookup.StaticVocabularyTable(
            tf.lookup.TextFileInitializer(
              embedding_file, tf.string, 0, tf.int64, tf.lookup.TextFileIndex.LINE_NUMBER, delimiter=' ',
            ),
            num_oov_buckets=1
          )
          # tf.lookup.index_table_from_file(embedding_file,
          #     num_oov_buckets=1,
          #     key_column_index=0,
          #     delimiter=' ')
          self.vocab_names_sizes[embeddings_name] = vocab_lookup_ops[embeddings_name].size()

    logging.log(logging.INFO, "Created %d vocab lookup ops: %s" %
                   (len(vocab_lookup_ops), str([k for k in vocab_lookup_ops.keys()])))
    return vocab_lookup_ops

  '''
  Gets the cached vocab ops for the given datafile, creating them if they already exist.
  This is needed in order to avoid re-creating duplicate lookup ops for each dataset input_fn, 
  since the lookup ops need to be called lazily from the input_fn in order to end up in the same tf.Graph.
  
  Args:
    word_embedding_file: (Optional) file containing word embedding vocab, with words in the first space-separated column
  
  Returns:
    Map from vocab names to tf.contrib.lookup ops.
    
  '''
  def get_lookup_ops(self, word_embedding_file=None):
    if self.vocab_lookups is None:
      self.vocab_lookups = self.create_vocab_lookup_ops(word_embedding_file)
    return self.vocab_lookups

  def create_load_or_update_vocab_files(self, data_config, save_dir, filenames=None, update_only=False):
    """
    Generates vocab files with counts for all the data with the vocab key
    set to True in data_config. Assumes the input file is in CoNLL format.

    Args:
      filename: Name of data file to generate vocab files from
      data_config: Data configuration map

    Returns:
      Map from vocab names to their sizes
    """

    # init maps
    vocabs = []
    vocabs_index = {}
    for d in data_config:
      updatable = data_config[d].get('updatable')
      if data_config[d].get('vocab') == d and (updatable or not update_only):
        this_vocab = {}
        if update_only and updatable and d in self.vocab_maps:
          this_vocab = self.vocab_maps[d]
        vocabs.append(this_vocab)
        vocabs_index[d] = len(vocabs_index)

    # Create vocabs from data files
    if filenames:
      for filename in filenames:
        with open(filename, 'r') as f:
          for line in f:
            line = line.strip()
            if line:
              split_line = line.split()
              for d in vocabs_index.keys():
                datum_idx = data_config[d]['conll_idx']
                this_vocab_map = vocabs[vocabs_index[d]]
                converter_name = data_config[d]['converter']['name'] if 'converter' in data_config[d] else 'default_converter'
                converter_params = data_converters.get_params(data_config[d], split_line, datum_idx)
                this_data = data_converters.dispatch(converter_name)(**converter_params)
                for this_datum in this_data:
                  if this_datum not in this_vocab_map:
                    this_vocab_map[this_datum] = 0
                  this_vocab_map[this_datum] += 1

    # Assume we have the vocabs saved to disk; load them
    else:
      for d in vocabs_index.keys():
        this_vocab_map = vocabs[vocabs_index[d]]
        with open("%s/%s.txt" % (self.vocabs_dir, d), 'r') as f:
          for line in f:
            datum, count = line.strip().split()
            this_vocab_map[datum] = int(count)

    # build reverse_maps, joint_label_lookup_maps
    for v in vocabs_index.keys():

      # build reverse_lookup map, from int -> string
      this_counts_map = vocabs[vocabs_index[v]]
      this_map = dict(zip(this_counts_map.keys(), range(len(this_counts_map.keys()))))
      reverse_map = dict(zip(range(len(this_counts_map.keys())), this_counts_map.keys()))
      self.oovs[v] = False
      if 'oov' in self.data_config[v] and self.data_config[v]['oov']:
        self.oovs[v] = True
        # reverse_map[len(reverse_map)] = constants.OOV_STRING
        # this_map[len(this_map)] = constants.OOV_STRING
      self.reverse_maps[v] = reverse_map
      self.vocab_maps[v] = this_map

      # check whether we need to build joint_label_lookup_map
      if 'label_components' in self.data_config[v]:
        joint_vocab_map = vocabs[vocabs_index[v]]  # vocabs["joint_pos_predicate"]
        label_components = self.data_config[v]['label_components']  # "gold_pos", "predicate"
        component_keys = [vocabs[vocabs_index[d]].keys() for d in label_components]  # list of vocab keys
        component_maps = [dict(zip(comp_keys, range(len(comp_keys)))) for comp_keys in component_keys]  # w2i
        map_names = ["%s_to_%s" % (v, label_comp) for label_comp in label_components]
        joint_to_comp_maps = [np.zeros([len(joint_vocab_map), 1], dtype=np.int32) for _ in label_components]
        for joint_idx, joint_label in enumerate(joint_vocab_map.keys()):
          split_label = joint_label.split(constants.JOINT_LABEL_SEP)
          for label_comp, comp_map, joint_to_comp_map in zip(split_label, component_maps, joint_to_comp_maps):
            comp_idx = comp_map[label_comp]
            joint_to_comp_map[joint_idx] = comp_idx

        # add them to the master map
        for map_name, joint_to_comp_map in zip(map_names, joint_to_comp_maps):
          self.joint_label_lookup_maps[map_name] = joint_to_comp_map

    for d in vocabs_index.keys():
      this_vocab_map = vocabs[vocabs_index[d]]
      with open("%s/%s.txt" % (self.vocabs_dir, d), 'w') as f:
        for k, v in this_vocab_map.items():
          print("%s\t%d" % (k, v), file=f)

    return {k: len(vocabs[vocabs_index[k]]) for k in vocabs_index.keys()}

  def make_vocab_files(self, data_config, save_dir, filenames=None):
    return self.create_load_or_update_vocab_files(data_config, save_dir, filenames, False)

  def update_vocab_files(self, filenames):
    vocab_names_sizes = self.create_load_or_update_vocab_files(self.data_config, self.save_dir, filenames, True)

    # merge new and old
    for vocab_name, vocab_size in vocab_names_sizes.items():
      self.vocab_names_sizes[vocab_name] = vocab_size
