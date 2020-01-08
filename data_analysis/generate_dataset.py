import argparse
import pickle
import pandas as pd

from data_analysis_utils import ner_tool

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate the updated dataset to be used for the experiments')
    parser.add_argument("--train-pickle", type=str, help="Path to the train pickle file.")
    parser.add_argument("--test-pickle", type=str, help="Path to the test pickle file.")
    parser.add_argument("--output-train", type=str, help="Output path of the update train dataset.")
    parser.add_argument("--output-test", type=str, help="Output path of the update train dataset.")
    parser.add_argument("--ner", type=str, help="Specify the NER tool to be used (none, spacy, custom-spacy)",
                        default="none")
    parser.add_argument("--kfold", type=int, help="If it is greater than 0, then we generate files for k-fold validation.",
                        default=0)

    # Parse the argument
    args = parser.parse_args()

    # Open the pickle files
    with (open(args.train_pickle, "rb")) as openfile:
        train = pickle.load(openfile)

    #with (open(args.test_pickle, "rb")) as openfile:
    #    test = pickle.load(openfile)

    # Iterate over the files and update accordingly
    train_result = []
    for index, row in train.iterrows():
        phrase, lemmas, pos, concepts, combined = ner_tool(row, method=args.ner)
        train_result.append([phrase, lemmas, pos, concepts, combined])

    print(pd.DataFrame(train_result, columns=["tokens", "lemmas", "pos", "concepts", "combined"]))

