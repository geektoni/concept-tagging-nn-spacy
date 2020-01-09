import argparse
import pickle

parser = argparse.ArgumentParser(description='Inspect pickle files used for training/testing')
parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the file we want to explore")
parser.add_argument("--save", default=False, action="store_true", help="Save the file to a manageable csv format")
args = parser.parse_args()

objects = []
with (open(args.filename[0], "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

print(objects[0].columns)
print(objects)

# Save pickle to csv file to be inspected
if args.save:
    objects[0].to_csv("saved_pickle.csv")
