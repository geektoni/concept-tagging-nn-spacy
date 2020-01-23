#
# This software is distributed under MIT license (see LICENSE file).
#
# Authors: Giovanni De Toni
#

import argparse
import pickle
import numpy as np
import pandas as pd


def _to_w2v_indexes(sentence, w2v_vocab):
    """
    Return which tokens in a sentence cannot be found inside the
    given embedding
    :param sentence: current sentence (tokenized)
    :param w2v_vocab: embedding matrix
    :return: a list of the tokens which were not found
    """
    total_missing = []
    for word in sentence:
        if word in w2v_vocab:
            pass
        elif word.title() in w2v_vocab:
            pass
        elif word.isdigit() or word.find("DIGIT") != -1:
            pass
        else:
            total_missing.append(word)
    return total_missing


parser = argparse.ArgumentParser(description='Inspect pickle files used for training/testing')
parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the file we want to explore")
parser.add_argument("--column", type=str, help="Print exactly this column")
parser.add_argument("--quick-bert-convert", default=False, action="store_true", help="Quickly convert BERT to correct file")
parser.add_argument("--save", default=False, action="store_true", help="Save the file to a manageable csv format")
parser.add_argument("--convert-to-bz2", default=False, help="Convert the given pickle to bz2", action="store_true")
parser.add_argument("--read-emb", default="../embedding.pickle", help="File with the embedding")
parser.add_argument("--copy-ner-from", type=str, help="Copy the ner column and add it to the current pickle", default="./train_updated.pickle")
parser.add_argument("--copy-ner-to", type=str, help="Path to the pickle file which will received the copied NER.")
args = parser.parse_args()

objects = []
with (open(args.filename[0], "rb")) as openfile:
    while True:
        try:
            objects.append(pd.read_pickle(openfile, compression="bz2"))
        except EOFError:
            break

# Print the columns of the file(s) and its content
for o in objects:
    print(o.columns)

if args.column:
    for o in objects:
        print(o[args.column])
else:
    print(objects)

# Print unique POS tag
total_pos = []
for o in objects:
    for index, row in o.iterrows():
        total_pos += row["pos"]
print(set(total_pos))

# Print how many sentences were NER tagged
total_tagged_ner = 0
total_ner = []
for o in objects:
    for index, row in o.iterrows():
        for emb in row["ner_enc"]:
            total_tagged_ner += sum(emb)
            total_ner.append("-".join([str(x) for x in emb]))
print(len(set(total_ner)))
print(total_tagged_ner)

# Try to convert each word with an embeddings and count how many I miss
emb = pd.read_pickle(args.read_emb)
vocab = emb["token"].unique()
print(len(vocab))
total_words = []
total_tokens = []
for index, row in objects[0].iterrows():
    total_words += _to_w2v_indexes(row["tokens"], vocab)
    total_tokens += row["tokens"]
print("Missing Words:", len(set(total_words)))
print("Coverage:", 1-len(set(total_words))/len(set(total_tokens)))


if args.quick_bert_convert:
    total_tokens = []
    for o in objects:
        for index, row in o.iterrows():
            raw_emb = []
            for e in row["tokens_emb"]:
                raw_emb.append(e[1][0])
            total_tokens.append(raw_emb)
    objects[0]["tokens_emb"] = np.array(total_tokens)

if args.copy_ner_to:

    print("Read input file")
    input = pd.read_pickle(args.copy_ner_from, compression="bz2")["ner_enc"]
    print("Read output file")
    output = pd.read_pickle(args.copy_ner_to, compression="bz2")

    print("Generate output")
    output["ner_enc"] = input
    output.to_pickle(args.copy_ner_to+"updated.bz2", compression="bz2")

if args.convert_to_bz2:
    for o in objects:
        o.to_pickle("./converted.bz2", compression="bz2")

# Save pickle to csv file(s) to be inspected
if args.save:
    for index, o in enumerate(objects):
        if not args.quick_bert_convert:
            o.to_csv("saved_pickle_{}.csv".format(index))
        else:
            with open("saved_pickle_{}.pickle".format(index), "wb") as output:
                pickle.dump(o, output)