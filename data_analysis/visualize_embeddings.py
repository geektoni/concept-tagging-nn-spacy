import argparse
import pickle
from sklearn.manifold import TSNE

import random

import numpy as np

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

font = {'size' : 13}
matplotlib.rc('font', **font)

selected_tokens = ["movie.name", "director.name", "actor.name", "producer.name"]

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def convert_with_emb(word, w2v_vocab):
    vectors = []
    if word.isdigit() or word.find("DIGIT") != -1:
        vector = w2v_vocab[w2v_vocab.token == "number"]["vector"].values
        vectors.append(vector[0])
    else:
        vector = w2v_vocab[w2v_vocab.token == word]["vector"].values
        if len(vector) > 0:
            vectors.append(vector[0])
        else:
            vector = w2v_vocab[w2v_vocab.token == word.title()]["vector"].values
            if len(vector) > 0:
                vectors.append(vector[0])
            else:
                vector = np.array([0.0 for i in range(0, len(w2v_vocab["vector"][0]))], dtype=np.dtype(float))
                vectors.append(vector)
    return np.array(vectors[0])

def tnse_plot(model, emb, random_=False):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    print("Reading the files")
    with tqdm(total=len(list(model.iterrows()))) as pbar:
        for k, e in model.iterrows():

            for i in range(0, len(e["tokens"])):
                value = e["concepts"][i].split("-")
                if value[0] != "O":
                    value = value[1]
                    if value in selected_tokens:
                        if emb is not None:
                            converted = convert_with_emb(e["tokens"][i], emb)
                            tokens.append(converted)
                        else:
                            tokens.append(e["tokens_emb"][i])
                        #labels.append(e["concepts"][i])
                        labels.append(value)

            pbar.update(1)

    labels_plot = list(set(labels))
    colormap_base = get_cmap(len(labels_plot), name="viridis")
    colormap = []
    for i in range(0, len(labels_plot)):
        colormap.append(matplotlib.colors.rgb2hex(colormap_base(i)[:3]))
    colormap = np.array(colormap)

    labels_value = []
    for l in labels:
        for i in range(0, len(labels_plot)):
            if l == labels_plot[i]:
                labels_value.append(i)
                break

    if not random_:
        print("Running TNSE")
        tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23, n_jobs=-1)
        new_values = tsne_model.fit_transform(tokens)
    else:
        new_values = []
        for t in tokens:
            new_values.append(
                (random.randrange(0, len(tokens)),
                 random.randrange(0, len(tokens))
                 )
            )


    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16))
    scatter = plt.scatter(x, y, c=colormap[np.array(labels_value)])

    handles = []
    for i in range(0, len(labels_plot)):
        pop_a = mpatches.Patch(color=colormap[i], label=labels_plot[i])
        handles.append(pop_a)

    plt.legend(handles=handles, loc="upper right", title="Concepts")

    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Produce visualizations of the embeddings.')
    parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the file we want to explore")
    parser.add_argument("--use-emb", type=str, help="Use custom embedding file")
    parser.add_argument("--random", action="store_true", help="Assign label randomly", default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Read embedding file
    emb = None
    if args.use_emb:
        with (open(args.use_emb, "rb")) as openfile:
            emb = pickle.load(openfile)

    # Read all the needed objects
    objects = []
    with (open(args.filename[0], "rb")) as openfile:
        while True:
            try:
                objects.append(pickle.load(openfile))
            except EOFError:
                break

    # Plot the visualization
    tnse_plot(objects[0], emb, random_=args.random)
