import argparse
import torch
from sklearn.manifold import TSNE

import random

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import seaborn as sns
sns.set(font_scale=1.3)

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

def convert_elmo_concepts_emb(x, W):
    return torch.sum(x * W[:, None, None], axis=0).numpy()

def tnse_plot(model, emb, random_=False, save=None, legend=True, elmo=False):
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
                            if elmo:
                                w = torch.tensor([1/3, 1/3, 1/3])
                                x = torch.FloatTensor(e["tokens_emb"])
                                tokens.append(
                                    convert_elmo_concepts_emb(x, w)[i]
                                )
                            else:
                                tokens.append(e["tokens_emb"][i])
                        labels.append(value)

            pbar.update(1)

    labels_plot = list(set(labels))
    colormap_base = get_cmap(len(labels_plot), name="viridis")
    colormap = []
    for i in range(0, len(labels_plot)):
        colormap.append(matplotlib.colors.rgb2hex(colormap_base(i)[:3]))
    colormap = np.array(colormap)

    labels_value = []
    labels_plot_colors = []
    for l in labels:
        for i in range(0, len(labels_plot)):
            if l == labels_plot[i]:
                labels_value.append(i)
                labels_plot_colors.append(labels_plot[i])
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

    if legend:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    data = pd.DataFrame(list(zip(x, y, labels_plot_colors)), columns=["x", "y", "Concept"])
    sns.set_palette(sns.color_palette("muted", len(labels_plot)))
    scatter = sns.scatterplot(data=data, x="x", y="y", hue="Concept", s=50)

    handles = []
    for i in range(0, len(labels_plot)):
        pop_a = mpatches.Patch(color=colormap[i], label=labels_plot[i])
        handles.append(pop_a)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        plt.legend( handles=handles[1:], labels=labels[1:],
                    loc="upper center",
                    bbox_to_anchor=(0., 1.02, 1., .102),
                    ncol=len(labels_plot))
    else:
        ax.legend_.remove()

    if save is not None:
        plt.tight_layout()
        plt.savefig(save, dpi=250)
    else:
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Produce visualizations of the embeddings.')
    parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the file we want to explore")
    parser.add_argument("--use-emb", type=str, help="Use custom embedding file")
    parser.add_argument("--random", action="store_true", help="Assign label randomly", default=False)
    parser.add_argument("--save", type=str, help="Save the image to disk")
    parser.add_argument("--no-legend", action="store_false", help="Plot also the legend", default=True)
    parser.add_argument("--elmo", action="store_true", help="Assign label randomly", default=False)

    # Parse the arguments
    args = parser.parse_args()

    # Read embedding file
    emb = None
    if args.use_emb:
        print("Read embedding file")
        emb = pd.read_pickle(args.use_emb, compression="infer")

    # Read all the needed objects
    objects = []
    print("Read files")
    for f in args.filename:
        objects.append(pd.read_pickle(f, compression="infer"))

    # Plot the visualization
    tnse_plot(objects[0], emb, random_=args.random, save=args.save, legend=args.no_legend, elmo=args.elmo)
