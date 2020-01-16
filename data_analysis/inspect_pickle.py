import argparse
import pickle
import numpy as np

parser = argparse.ArgumentParser(description='Inspect pickle files used for training/testing')
parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the file we want to explore")
parser.add_argument("--column", type=str, help="Print exactly this column")
parser.add_argument("--quick-bert-convert", default=False, action="store_true", help="Quickly convert BERT to correct file")
parser.add_argument("--save", default=False, action="store_true", help="Save the file to a manageable csv format")
parser.add_argument("--convert-to-bz2", default=False, action="store_true")
args = parser.parse_args()

objects = []
with (open(args.filename[0], "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
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


if args.quick_bert_convert:
    total_tokens = []
    for o in objects:
        for index, row in o.iterrows():
            raw_emb = []
            for e in row["tokens_emb"]:
                raw_emb.append(e[1][0])
            total_tokens.append(raw_emb)
    objects[0]["tokens_emb"] = np.array(total_tokens)

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