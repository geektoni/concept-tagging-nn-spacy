#
# This software is distributed under MIT license (see LICENSE file).
#
# Authors: Giovanni De Toni
#

import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Convert the GloVe embeddings into the correct format')
parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the GloVE embedding file")

args = parser.parse_args()

for f in args.filename:
    objects = []
    with (open(f, "rb")) as openfile:

        # For each line inside the file, we convert it into
        # the format required by our project.
        for l in openfile:
            l = l.decode("utf-8").replace("\n", "")
            l = l.split(" ")
            emb = [float(l[i]) for i in range(1, len(l))]
            new_obj = [l[0], emb]
            objects.append(new_obj)

        df = pd.DataFrame(objects, columns=["token", "vector"])
        df.to_pickle(f+".updated.bz2", compression="bz2")

