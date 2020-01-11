import argparse
import pickle
import pandas as pd

from data_analysis_utils import ner_tool, one_hot_encoding_pos, one_hot_encoding_ner

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate the updated dataset to be used for the experiments')
    parser.add_argument("--train-pickle", type=str, help="Path to the train pickle file.")
    parser.add_argument("--test-pickle", type=str, help="Path to the test pickle file.")
    parser.add_argument("--output-train", type=str, help="Output path of the update train dataset.",
                        default="./train_updated.pickle")
    parser.add_argument("--output-test", type=str, help="Output path of the update train dataset.",
                        default="./test_updated.pickle")
    parser.add_argument("--ner", type=str, help="Specify the NER tool to be used (none, spacy, custom-spacy)",
                        default="spacy")
    parser.add_argument("--replace", type=str, help="Replace the O tag with a more informative value (keep, stem, lemma, word)",
                        default="keep")
    parser.add_argument("--kfold", type=int, help="If it is greater than 0, then we generate files for k-fold validation.",
                        default=0)
    parser.add_argument("--save", default="False", action="store_true", help="Save the dataset to file.")
    parser.add_argument("--verbose", default="False", action="store_true", help="Print more information")

    # Parse the argument
    args = parser.parse_args()

    # Open the pickle files
    with (open(args.train_pickle, "rb")) as openfile:
        train = pickle.load(openfile)

    with (open(args.test_pickle, "rb")) as openfile:
        test = pickle.load(openfile)

    # Iterate over the train file
    train_result = []
    for index, row in train.iterrows():
        phrase, lemmas, pos, concepts, ner, combined, = ner_tool(row, method=args.ner,
                                                           replace_O=args.replace)
        pos_enc = one_hot_encoding_pos(pos)
        ner_enc = one_hot_encoding_ner(ner)
        train_result.append([phrase, lemmas, pos, pos_enc, concepts, ner_enc, combined])


    # Iterate over the test file
    test_result = []
    for index, row in test.iterrows():
        phrase, lemmas, pos, concepts, ner, combined, = ner_tool(row, method=args.ner,
                                                                 replace_O=args.replace)
        pos_enc = one_hot_encoding_pos(pos)
        ner_enc = one_hot_encoding_ner(ner)
        test_result.append([phrase, lemmas, pos, pos_enc, concepts, ner_enc, combined])

    train_updated = pd.DataFrame(train_result, columns=["tokens", "lemmas", "pos", "pos_enc", "concepts", "ner_enc", "combined"])
    test_updated = pd.DataFrame(test_result, columns=["tokens", "lemmas", "pos", "pos_enc", "concepts", "ner_enc", "combined"])

    if args.verbose:
        print(train_updated.columns)
        print(train_updated)
        print(test_updated.columns)
        print(test_updated)

    # Save the result dataframes into a pickle format
    if args.save:
        train_updated.to_pickle(args.output_train)
        test_updated.to_pickle(args.output_test)