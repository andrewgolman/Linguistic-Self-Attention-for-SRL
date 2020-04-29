import data_converters


def conll_data_generator(filenames, data_config):
  """
  AG: Yields sentences as lists of words with features
  """
  for filename in filenames:
    with open(filename, 'r') as f:
      sents = 0
      toks = 0
      buf = []
      for line in f:
        line = line.strip()
        if line:
          toks += 1
          split_line = line.split()
          data_vals = []
          for d in data_config.keys():
            # only return the data that we're actually going to use as inputs or outputs
            if data_config[d].get('feature') or data_config[d].get('label'):
              datum_idx = data_config[d]['conll_idx']
              converter_name = data_config[d]['converter']['name'] if 'converter' in data_config[d] else 'default_converter'
              converter_params = data_converters.get_params(data_config[d], split_line, datum_idx)
              data = data_converters.dispatch(converter_name)(**converter_params)
              data_vals.extend(data)

          buf.append(tuple(data_vals))
        else:
          if buf:
            sents += 1
            yield buf
            buf = []
      # catch the last one
      if buf:
        yield buf
