import argparse
import pickle
import spacy

from spacy.scorer import Scorer
from spacy.gold import GoldParse
from spacy.util import minibatch
from spacy.util import compounding

import random
from pathlib import Path

from tqdm import tqdm

def convert_dataset_into_spacy(df):
    data = []

    for index, row in df.iterrows():
        tokens, concepts = row["tokens"], row["concepts"]

        entities = []
        total_words = 0

        for i in range(0, len(tokens)):
            if concepts[i] != "O":
                entities.append((total_words, total_words+len(tokens[i])-1, concepts[i]))
            total_words += len(concepts)+1 # Add the length of the current token plus a space

        data.append(
            (" ".join(tokens),
             {"entities": entities}
             )
        )

    return data

def evaluate(model, examples):
  scorer = Scorer()

  for input_, annot in examples:

    doc_gold_text = model.make_doc(input_)
    gold = GoldParse(doc_gold_text, entities=annot['entities'])
    pred_value = model(input_)
    scorer.score(pred_value, gold)

  return scorer.scores

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a custom spaCy NER classifier.")
    parser.add_argument("--train-pickle", type=str, help="Path to the train file (pickle format)")
    parser.add_argument("--test-pickle", type=str, help="Path to the test file (pickle format)")
    parser.add_argument("--model", type=str, help="Train the NER with a default spaCy model (en_core_web_sm).")
    parser.add_argument("--iterations", type=int, help="Training iterations", default=100)
    parser.add_argument("--output", type=str, help="Directory where the trained NER will be saved.")

    args = parser.parse_args()

    with (open(args.train_pickle, "rb")) as openfile:
        train = pickle.load(openfile)

    with (open(args.test_pickle, "rb")) as openfile:
        test = pickle.load(openfile)

    # Generate the data in the correct format
    train = convert_dataset_into_spacy(train)
    test = convert_dataset_into_spacy(test)

    # Create the blank NER model or start from an already trained one
    if args.model is not None:
        nlp = spacy.load(args.model)  # load existing spaCy model
        print("Loaded model '%s'" % args.model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    else:
        ner = nlp.get_pipe("ner")

    # Add the needed labels from the train dataset
    for _, annotations in train:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    other_pipe = [pipe for pipe in nlp.pipe_names if pipe != 'ner']

    # Only training NER
    with nlp.disable_pipes(*other_pipe):
        if args.model is None:
            optimizer = nlp.begin_training()
        else:
            optimizer = nlp.resume_training()

    # Start the training procedure
    for iter in tqdm(range(args.iterations)):

        print(" [*] Starting iteration " + str(iter))

        random.shuffle(train)
        losses = {}

        batches = minibatch(train, size=compounding(4., 32., 1.001))

        for batch in tqdm(batches):
            texts, annotations = zip(*batch)
            nlp.update(
                texts,
                annotations,
                drop=0.2,
                sgd=optimizer,
                losses=losses
            )

    new_model = nlp

    # Evaluate the model on the test data
    test_result = evaluate(new_model, test)

    # Print the test results
    print(test_result)

    # Save model
    if args.output is not None:
        output_dir = Path(args.output)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta['name'] = "movie_en_ner"  # rename model
        nlp.to_disk(output_dir)
        print("[*] Saved model to", output_dir)
