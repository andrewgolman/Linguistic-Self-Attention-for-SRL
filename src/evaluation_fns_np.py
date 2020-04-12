import numpy as np
import util
import os
from subprocess import check_output, CalledProcessError
import tensorflow.compat.v1.logging as logging


# todo simplify to convert_bio
def convert_bilou(bio_predicted_roles):
  '''

  :param bio_predicted_roles: sequence of BIO-encoded predicted role labels
  :return: sequence of conll-formatted predicted role labels
  '''

  converted = []
  started_types = []
  for i, s in enumerate(bio_predicted_roles):
    s = s if isinstance(s, str) else s.decode('utf-8')
    label_parts = s.split('/')
    curr_len = len(label_parts)
    combined_str = ''
    Itypes = []
    Btypes = []
    for idx, label in enumerate(label_parts):
      bilou = label[0]
      label_type = label[2:]
      props_str = ''
      if bilou == 'I':
        Itypes.append(label_type)
        props_str = ''
      elif bilou == 'O':
        curr_len = 0
        props_str = ''
      elif bilou == 'U':
        # need to check whether last one was ended
        props_str = '(' + label_type + ('*)' if idx == len(label_parts) - 1 else "")
      elif bilou == 'B':
        # need to check whether last one was ended
        props_str = '(' + label_type
        started_types.append(label_type)
        Btypes.append(label_type)
      elif bilou == 'L':
        props_str = ')'
        started_types.pop()
        curr_len -= 1
      combined_str += props_str
    while len(started_types) > curr_len:
      converted[-1] += ')'
      started_types.pop()
    while len(started_types) < len(Itypes) + len(Btypes):
      combined_str = '(' + Itypes[-1] + combined_str
      started_types.append(Itypes[-1])
      Itypes.pop()
    if not combined_str:
      combined_str = '*'
    elif combined_str[0] == "(" and combined_str[-1] != ")":
      combined_str += '*'
    elif combined_str[-1] == ")" and combined_str[0] != "(":
      combined_str = '*' + combined_str
    converted.append(combined_str)
  while len(started_types) > 0:
    converted[-1] += ')'
    started_types.pop()
  return converted


def convert_conll(predicted_roles):
  '''

  :param bio_predicted_roles: sequence of predicted role labels
  :return: sequence of conll-formatted predicted role labels
  '''

  def convert_single(s):
    s = s if isinstance(s, str) else s.decode('utf-8')
    return "*" if s == "_" else "(%s*)" % s

  converted = map(convert_single, predicted_roles)
  return converted


# Write targets file w/ format:
# -        (A1*  (A1*
# -          *     *
# -          *)    *)
# -          *     *
# expected (V*)    *
# -        (C-A1*  *
# widen     *     (V*)
# -         *     (A4*
def write_srl_eval(filename, words, predicates, sent_lens, role_labels, first_dim_from_batch=False):
  with open(filename, 'w') as f:
    role_labels_start_idx = 0
    num_predicates_per_sent = np.sum(predicates, -1)

    words = util.batch_str_decode(words)

    # for each sentence in the batch
    for sent_id in range(words.shape[0]):
        sent_words = words[sent_id]
        sent_predicates = predicates[sent_id]
        sent_len = sent_lens[sent_id]
        sent_num_predicates = num_predicates_per_sent[sent_id]


        # grab predicates and convert to conll format from bio
        if first_dim_from_batch:
            sent_role_labels_bio = role_labels[sent_id].transpose()
        else:
            # this is a sent_num_predicates x batch_seq_len array
            sent_role_labels_bio = role_labels[role_labels_start_idx : role_labels_start_idx + sent_num_predicates]

        # this is a list of sent_num_predicates lists of srl role labels
        sent_role_labels = list(map(list, zip(*[convert_bilou(j[:sent_len]) for j in sent_role_labels_bio])))
        role_labels_start_idx += sent_num_predicates

        # for each token in the sentence
        for j, (word, predicate) in enumerate(zip(sent_words[:sent_len], sent_predicates[:sent_len])):
            tok_role_labels = sent_role_labels[j] if sent_role_labels else []
            predicate_str = word if predicate else '-'
            roles_str = '\t'.join(tok_role_labels)
            print("%s\t%s" % (predicate_str, roles_str), file=f)
        print(file=f)


# Write to this format for eval.pl:
# 1       The             _       DT      _       _       2       det
# 2       economy         _       NN      _       _       4       poss
# 3       's              _       POS     _       _       2       possessive
# 4       temperature     _       NN      _       _       7       nsubjpass
# 5       will            _       MD      _       _       7       aux
def write_parse_eval(filename, words, parse_heads, sent_lens, parse_labels, pos_tags):

  words = util.batch_str_decode(words)
  pos_tags = util.batch_str_decode(pos_tags)
  parse_labels = util.batch_str_decode(parse_labels)

  with open(filename, 'w') as f:

    # for each sentence in the batch
    for sent_words, sent_parse_heads, sent_len, sent_parse_labels, sent_pos_tags in zip(words, parse_heads, sent_lens,
                                                                                        parse_labels, pos_tags):
      # for each token in the sentence
      for j, (word, parse_head, parse_label, pos_tag) in enumerate(zip(sent_words[:sent_len],
                                                                       sent_parse_heads[:sent_len],
                                                                       sent_parse_labels[:sent_len],
                                                                       sent_pos_tags[:sent_len])):
        parse_head = 0 if j == parse_head else parse_head + 1
        print("%d\t%s\t_\t%s\t_\t_\t%d\t%s" % (j, word, pos_tag, int(parse_head), parse_label), file=f)
      print(file=f)


def write_srl_debug(filename, words, predicates, sent_lens, role_labels, pos_predictions, pos_targets):
  with open(filename, 'w') as f:
    role_labels_start_idx = 0
    num_predicates_per_sent = np.sum(predicates, -1)
    # for each sentence in the batch
    for sent_words, sent_predicates, sent_len, sent_num_predicates, pos_preds, pos_targs in zip(words, predicates, sent_lens,
                                                                          num_predicates_per_sent, pos_predictions,
                                                                          pos_targets):
      # grab predicates and convert to conll format from bio
      # this is a sent_num_predicates x batch_seq_len array
      sent_role_labels_bio = role_labels[role_labels_start_idx: role_labels_start_idx + sent_num_predicates]

      # this is a list of sent_num_predicates lists of srl role labels
      sent_role_labels = list(map(list, zip(*[convert_bilou(j[:sent_len]) for j in sent_role_labels_bio])))
      role_labels_start_idx += sent_num_predicates

      sent_role_labels_bio = list(zip(*sent_role_labels_bio))

      pos_preds = list(map(lambda d: d.decode('utf-8'), pos_preds))
      pos_targs = list(map(lambda d: d.decode('utf-8'), pos_targs))

      # for each token in the sentence
      # printed = False
      for j, (word, predicate, pos_p, pos_t) in enumerate(zip(sent_words[:sent_len], sent_predicates[:sent_len],
                                                              pos_preds[:sent_len], pos_targs[:sent_len])):
        tok_role_labels = sent_role_labels[j] if sent_role_labels else []
        bio_tok_role_labels = sent_role_labels_bio[j][:sent_len] if sent_role_labels else []
        word_str = word.decode('utf-8')
        predicate_str = str(predicate)
        roles_str = '\t'.join(tok_role_labels)
        bio_roles_str = '\t'.join(map(lambda d: d.decode('utf-8'), bio_tok_role_labels))
        print("%s\t%s\t%s\t%s\t%s\t%s" % (word_str, predicate_str, pos_t, pos_p, roles_str, bio_roles_str), file=f)
      print(file=f)


def conll_srl_eval(srl_predictions, predicate_predictions, words, mask, srl_targets, predicate_targets,
                      pred_srl_eval_file, gold_srl_eval_file, pos_predictions=None, pos_targets=None):

  # predictions: num_predicates_in_batch x batch_seq_len tensor of ints
  # predicate predictions: batch_size x batch_seq_len [ x 1?] tensor of ints (0/1)
  # words: batch_size x batch_seq_len tensor of ints (0/1)

  # need to print for every word in every sentence
  sent_lens = np.sum(mask, -1).astype(np.int32)

  # import time
  # debug_fname = pred_srl_eval_file.decode('utf-8') + str(time.time())
  # write_srl_debug(debug_fname, words, predicate_targets, sent_lens, srl_targets, pos_predictions, pos_targets)

  # write gold labels
  write_srl_eval(gold_srl_eval_file, words, predicate_targets, sent_lens, srl_targets, first_dim_from_batch=True)

  # write predicted labels
  write_srl_eval(pred_srl_eval_file, words, predicate_predictions, sent_lens, srl_predictions)

  # run eval script
  correct, excess, missed = 0, 0, 0
  with open(os.devnull, 'w') as devnull:
    try:
      srl_eval = check_output(["perl", "bin/srl-eval.pl", gold_srl_eval_file, pred_srl_eval_file], stderr=devnull)
      srl_eval = srl_eval.decode('utf-8')
      # print(srl_eval)
      correct, excess, missed = map(int, srl_eval.split('\n')[6].split()[1:4])
    except CalledProcessError as e:
      logging.log(logging.ERROR, "Call to srl-eval.pl (conll srl eval) failed.")

  return correct, excess, missed


def conll_parse_eval(parse_label_predictions, parse_head_predictions, words, mask, parse_label_targets,
                        parse_head_targets, pred_eval_file, gold_eval_file, pos_targets):

  # need to print for every word in every sentence
  sent_lens = np.sum(mask, -1).astype(np.int32)

  # write gold labels
  write_parse_eval(gold_eval_file, words, parse_head_targets, sent_lens, parse_label_targets, pos_targets)

  # write predicted labels
  write_parse_eval(pred_eval_file, words, parse_head_predictions, sent_lens, parse_label_predictions, pos_targets)

  # run eval script
  total, labeled_correct, unlabeled_correct, label_correct = 0, 0, 0, 0
  with open(os.devnull, 'w') as devnull:
    try:
      eval = check_output(["perl", "bin/eval.pl", "-g", gold_eval_file, "-s", pred_eval_file], stderr=devnull)
      eval_str = eval.decode('utf-8')

      # Labeled attachment score: 26444 / 29058 * 100 = 91.00 %
      # Unlabeled attachment score: 27251 / 29058 * 100 = 93.78 %
      # Label accuracy score: 27395 / 29058 * 100 = 94.28 %
      first_three_lines = eval_str.split('\n')[:3]
      total = int(first_three_lines[0].split()[5])
      labeled_correct, unlabeled_correct, label_correct = map(lambda l: int(l.split()[3]), first_three_lines)
    except CalledProcessError as e:
      logging.log(logging.ERROR, "Call to eval.pl (conll parse eval) failed.")

  return total, np.array([labeled_correct, unlabeled_correct, label_correct])


def conll_srl_eval_np(predictions, targets, predicate_predictions, words, mask, predicate_targets, reverse_maps,
                   gold_srl_eval_file, pred_srl_eval_file, pos_predictions, pos_targets, accumulator):

  # first, use reverse maps to convert ints to strings
  str_srl_predictions = [list(map(reverse_maps['srl'].get, s)) for s in predictions]
  str_words = [list(map(reverse_maps['word'].get, s)) for s in words]
  str_srl_targets = [list(map(reverse_maps['srl'].get, s)) for s in targets]

  correct, excess, missed = conll_srl_eval(str_srl_predictions, predicate_predictions, str_words, mask, str_srl_targets,
                                           predicate_targets, pred_srl_eval_file, gold_srl_eval_file)

  accumulator['correct'] += correct
  accumulator['excess'] += excess
  accumulator['missed'] += missed

  precision = accumulator['correct'] / (accumulator['correct'] + accumulator['excess'])
  recall = accumulator['correct'] / (accumulator['correct'] + accumulator['missed'])
  f1 = 2 * precision * recall / (precision + recall)

  return f1


def conll_parse_eval_np(predictions, targets, parse_head_predictions, words, mask, parse_head_targets, reverse_maps,
                        gold_parse_eval_file, pred_parse_eval_file, pos_targets, accumulator):

  # first, use reverse maps to convert ints to strings
  str_words = [list(map(reverse_maps['word'].get, s)) for s in words]
  str_predictions = [list(map(reverse_maps['parse_label'].get, s)) for s in predictions]
  str_targets = [list(map(reverse_maps['parse_label'].get, s)) for s in targets]
  str_pos_targets = [list(map(reverse_maps['gold_pos'].get, s)) for s in pos_targets]

  total, corrects = conll_parse_eval(str_predictions, parse_head_predictions, str_words, mask, str_targets,
                                     parse_head_targets, pred_parse_eval_file, gold_parse_eval_file, str_pos_targets)

  accumulator['total'] += total
  accumulator['corrects'] += corrects

  accuracies = accumulator['corrects'] / accumulator['total']

  return accuracies
