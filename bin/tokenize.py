import argparse
import transformers


tokenizers = {
    'albert': transformers.AlbertTokenizer.from_pretrained('albert-base-v1'),
    't5': transformers.T5Tokenizer.from_pretrained('t5-base'),
    'bert': transformers.BertTokenizer.from_pretrained('bert-base-cased'),
}


def get_sentence(lines):
    return " ".join(l.split()[3] for l in lines)


def tokenize_words(input_file, output_file, tokenizer, multitoken=True):
    """
    # INPUT FORMAT
    # https://github.com/strubell/preprocess-conll05
    # Field 0: domain placeholder
    # Field 1: sentence id
    # Field 2: token id
    # Field 3: word form
    # Field 4: gold part-of-speech tag
    # Field 5: auto part-of-speech tag
    # Field 6: dependency parse head
    # Field 7: dependency parse label
    # Field 8: placeholder
    # Field 9: verb sense
    # Field 10: predicate (infinitive form)
    # Field 11: placeholder -> tokenized id
    # Field 12: placeholder -> word start (bool)
    # Field 13: NER placeholder
    # Fields range(14, 14+PRED_COUNT): for each predicate, a column representing the labeled arguments of the predicate.
    """
    with open(input_file) as fin, open(output_file, "w") as fout:
        sentences = set()
        lines = []
        ignored = 0
        total = 0
        token_count = 0

        for line in fin:
            if not line.strip():
                total += 1
                if token_count < 100 and len(lines[0].split()) > 14 and get_sentence(lines) not in sentences:
                    sentences.add(get_sentence(lines))
                    for s in lines:
                        print(s, file=fout)
                    print("", file=fout)
                else:
                    # print(len(lines), len(lines[0].split()), get_sentence(lines) in sentences)
                    ignored += 1
                lines = []
                token_count = 0
            else:
                fields = line.strip().split("\t")
                tokens = tokenizer.encode(fields[3], add_special_tokens=False)
                token_count += len(tokens)
                for i, t in enumerate(tokens):
                    if i == 0:
                        fields[12] = "1"
                    else:
                        for j in range(4, len(fields)):
                            fields[j] = "0"

                    fields[11] = str(t)
                    line = "\t".join(fields)
                    lines.append(line)

                    if not multitoken:
                        break

        print(input_file, "Ignored", ignored, "sentences from", total)


def main(file):
    print("INFO: tokenizing words in file", file)
    tokenizer = tokenizers['albert']  # todo AG add arg

    res_file = ".".join(file.split(".")[:-1] + ["single_albert"])
    tokenize_words(file, res_file, tokenizer, multitoken=False)

    # res_file = ".".join(file.split(".")[:-1] + ["lisa_albert"])
    # tokenize_words(file, res_file, tokenizer, multitoken=True)


if __name__ == "__main__":
    # arg_parser = argparse.ArgumentParser(description='One line for every word -> one line for every token')
    # arg_parser.add_argument('--in_file_name', type=str, help='File to process')
    # args = arg_parser.parse_args()
    # main(args.in_file_name)
    files = [
        "data/conll2012-train.txt.lisa",
        # "data/conll2012-dev.txt.lisa",
        "data/conll2012-test.txt.lisa",
        "data/conll2012-train-eval.txt.lisa",
    ]
    for file in files:
        main(file)
