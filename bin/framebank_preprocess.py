import json
import pickle
import pandas as pd

with open("data/cleared_corpus.json") as f:
    data = json.load(f)


with open("data/ru_srl_only.lisa", "w") as fout:
    for sid, sentence in data.groupby("ex_id"):
        words = sentence.tokens.values[0]
        predicates = sorted(list(set(sentence.prd_address)))
        head = {}
        args = {}
        syntax = {}
        for _, row in sentence.iterrows():
            for pred in predicates:
                args[(row.arg_address, pred)] = "O"
            head[row.arg_address] = row.prd_address
            args[(row.arg_address, row.prd_address)] = "_".join(row.role.split())
            syntax[row.arg_address] = row.syn_link_name

        s_conll = []
        for i, w in enumerate(words):
            s_conll.append([
                "fb", # Field 0: domain placeholder
                sid, # Field 1: sentence id
                i, # Field 2: token id
                w, # Field 3: word form
                "-", # Field 4: gold part-of-speech tag
                "-", # Field 5: auto part-of-speech tag,
                head.get(i, i), # Field 6: dependency parse head
                syntax.get(i, "-"), # Field 7: dependency parse label
                "-", # Field 8: placeholder
                "1" if i in predicates else "-", # Field 9: is_predicate
                "-", # Field 10: predicate (infinitive form)
                "-", # Field 11: placeholder -> tokenized id
                "-", # Field 12: placeholder -> word start (bool)
                "-", # Field 13: NER placeholder
                *[args.get((i, p), "O") for p in predicates],
                # Fields range(14, 14+PRED_COUNT): for each predicate, a column representing the labeled arguments of the predicate.
            ])
        for line in s_conll:
            print("\t".join(str(s) for s in line), file=fout)
        print("", file=fout)


with open("data/ling_data.pckl", "rb") as f:
    data = pickle.load(f)

struct_data = {}
for k, v in data:
    if k not in struct_data:
        struct_data[k] = v
    else:
        print(k, v['text'], len(v['sentences']))

count = 0


def srl_mask(line):
    srl_tags = line[14:]
    if line[9] == "1":
        return 1
    for t in srl_tags:
        if t != "O":
            return 1
    return 0


with open("data/ru_srl_only.lisa") as f:
    with open("data/ru_srl.lisa", 'w') as fout:
        lines = []
        idn = 0
        for line in f:
            if not line.strip():
                tokens = [line[3] for line in lines]
                base_tokens = struct_data[idx]['tokens']
                for sid, sentence in enumerate(struct_data[idx]['sentences']):
                    b, e = sentence.begin, sentence.end
                    this_sent_tokens = [base_tokens[i].text for i in range(b, e)]
                    if tokens == this_sent_tokens:
                        break
                else:
                    print("UNABLE TO FIND SENTENCE", idx, " ".join(tokens))
                    lines = []
                    continue

                pos_tag = [t if t else "-" for t in struct_data[idx]['postag'][sid]]
                tree = struct_data[idx]['syntax_dep_tree'][sid]
                parse_label = [node.link_name for node in tree]
                parse_head = [node.parent + 1 for i, node in enumerate(tree)]
                assert len(pos_tag) == len(tokens) == len(tree)  # , len(pos)
                vcount = 0
                for i in range(len(lines)):
                    lines[i][4] = pos_tag[i]
                    lines[i][5] = pos_tag[i]
                    lines[i][6] = parse_head[i]
                    lines[i][7] = parse_label[i]
                    lines[i][8] = srl_mask(lines[i])
                    if pos_tag[i] in ['VERB', 'PRED']:
                        lines[i][10] = "1"
                    else:
                        lines[i][10] = "-"
                    if lines[i][9] == "1":
                        lines[i][14 + vcount] = "V"
                        vcount += 1

                    for j in range(14, len(lines[i])):
                        if lines[i][j] != "O":
                            lines[i][j] = "B-" + lines[i][j]
                for line in lines:
                    print("\t".join(str(s) for s in line), file=fout)
                print("", file=fout)

                lines = []

                count += 1
                if count % 1000 == 0:
                    print(count)
            else:
                idx = line.split()[1]
                lines.append(line.strip().split())
