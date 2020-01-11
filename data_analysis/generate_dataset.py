import argparse
import pickle
import pandas as pd

from data_analysis_utils import ner_tool, one_hot_encoding_pos

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate the updated dataset to be used for the experiments')
    parser.add_argument("--train-pickle", type=str, help="Path to the train pickle file.")
    parser.add_argument("--test-pickle", type=str, help="Path to the test pickle file.")
    parser.add_argument("--output-train", type=str, help="Output path of the update train dataset.",
                        default="./train_updated.pickle")
    parser.add_argument("--output-test", type=str, help="Output path of the update train dataset.",
                        default="./test_updated.pickle")
    parser.add_argument("--ner", type=str, help="Specify the NER tool to be used (none, spacy, custom-spacy)",
                        default="none")
    parser.add_argument("--replace", type=str, help="Replace the O tag with a more informative value (keep, stem, lemma, word)",
                        default="keep")
    parser.add_argument("--kfold", type=int, help="If it is greater than 0, then we generate files for k-fold validation.",
                        default=0)
    parser.add_argument("--save", default="False", action="store_true", help="Save the dataset to file.")

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
                                                           replace_O=args.replace,
                                                           add_ner_feature=args.add_ner_feature)
        pos_enc = one_hot_encoding_pos(pos)


        train_result.append([phrase, lemmas, pos, pos_enc, concepts, ner, combined])


    # Iterate over the test file
    test_result = []
    for index, row in test.iterrows():
        phrase, lemmas, pos, concepts, ner, combined, = ner_tool(row, method=args.ner,
                                                                 replace_O=args.replace,
                                                                 add_ner_feature=args.add_ner_feature)
        pos_enc = one_hot_encoding_pos(pos)

        test_result.append([phrase, lemmas, pos, pos_enc, concepts, ner, combined])


    train_updated = pd.DataFrame(train_result, columns=["tokens", "lemmas", "pos", "pos_enc", "concepts", "ner", "combined"])
    test_updated = pd.DataFrame(test_result, columns=["tokens", "lemmas", "pos", "pos_enc", "concepts", "ner", "combined"])


    if args.save:
        train_updated.to_pickle(args.output_train)
        test_updated.to_pickle(args.output_test)