import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Convert the ConceptNet embeddings into the correct format')
parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the ConceptNet embedding file")

args = parser.parse_args()

for f in args.filename:
    objects = []
    with (open(f, "rb")) as openfile:

        # For each line inside the file, we convert it into
        # the format required by our project. The lines are
        # hardcoded
        with tqdm(total=516783) as pbar:
            skip_first_line = True
            for l in openfile:

                # Skip the first line
                if skip_first_line:
                    skip_first_line=False
                    continue

                l = l.decode("utf-8").replace("\n", "")
                l = l.split(" ")
                emb = [float(l[i]) for i in range(1, len(l))]
                new_obj = [l[0], emb]
                objects.append(new_obj)
                pbar.update(1)

        df = pd.DataFrame(objects, columns=["token", "vector"])
        df.to_pickle(f+".updated.pickle", compression="bz2")

