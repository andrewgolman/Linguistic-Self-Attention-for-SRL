import argparse


def reformat_bio_file(input_file, output_file):
    """
    # INPUT FORMAT
    # https://github.com/iesl/conll2012-preprocess-parsing + bio convertation at
    # https://github.com/iesl/conll2012-preprocess-parsing/blob/master/bin/convert-bio.sh
    # Field 0: domain placeholder
    # Field 1: sentence id
    # Field 2: token id
    # Field 3: word form
    # Field 4: gold part-of-speech tag
    # Field 5: auto part-of-speech tag
    # Field 6: dependency parse head
    # Field 7: dependency parse label
    # Field 8: placeholder
    # Field 9: predicate (infinitive form)
    # Field 10: verb sense
    # Field 11: ? (disregarding)
    # Field 12: ? (disregarding)
    # Field 13: NER placeholder
    # Fields range(14, 14+PRED_COUNT): for each predicate, a column representing the labeled arguments of the predicate.
    # Field -1: ? (disregarding)

    # OUTPUT FORMAT
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
    # Field 11: placeholder
    # Field 12: placeholder
    # Field 13: NER placeholder
    # Fields range(14, 14+PRED_COUNT): for each predicate, a column representing the labeled arguments of the predicate.
    """
    with open(input_file) as fin, open(output_file, "w") as fout:
        for line in fin:
            if not line.strip():
                print("", file=fout)
                continue

            fields = line.strip().split("\t")
            fields[9], fields[10] = fields[10], fields[9]
            fields.pop(-1)
            for i in range(14, len(fields)):
                fields[i] = fields[i].split("/")[0]
                fields[i] = fields[i].replace("C-", "").replace("R-", "")

            line = "\t".join(fields)
            print(line, file=fout)


def main(file):
    print("INFO: reformating fields in file", file)
    res_file = file[:-4]+".lisa"
    reformat_bio_file(file, res_file)


if __name__ == "__main__":

    arg_parser = argparse.ArgumentParser(description='Adjust format from the repository')
    arg_parser.add_argument('--in_file_name', type=str, help='File to process')
    args = arg_parser.parse_args()
    main(args.in_file_name)
