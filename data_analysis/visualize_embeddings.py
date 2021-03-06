#
# This software is distributed under MIT license (see LICENSE file).
#
# Authors: Giovanni De Toni
#

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

from mpl_toolkits.mplot3d import Axes3D

font = {'size' : 13}
matplotlib.rc('font', **font)

selected_tokens = ["movie.name"]

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

def tnse_plot(model, emb, random_=False, save=None, legend=True, elmo=False, D=False, facet=False):
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
                    #if True:
                        if emb is not None:
                            converted = convert_with_emb(e["tokens"][i], emb)
                            tokens.append(converted)
                        else:
                            if elmo:
                                #w = torch.tensor([0.4030, 0.2430, 0.0301])
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
    colormap_base = get_cmap(len(labels_plot), name="prism")
    colormap = []
    for i in range(0, len(labels_plot)):
        colormap.append(matplotlib.colors.rgb2hex(colormap_base(i)[:3]))
    colormap = np.array(colormap)

    labels_value = []
    labels_plot_colors = []
    D_palette = sns.color_palette("muted", len(labels_plot))

    for l in labels:
        for i in range(0, len(labels_plot)):
            if l == labels_plot[i]:
                labels_value.append(i)
                labels_plot_colors.append(labels_plot[i])
                break

    if not random_:
        print("Running TNSE")
        tsne_model = TSNE(perplexity=50, n_components=2 if not D else 3,
                          init='pca', n_iter=5000, random_state=23, n_jobs=-1)
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
    z = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        if D:
            z.append(value[2])

    # Get centroid and comput SSE
    centroid = (sum(x)/len(x), sum(y)/len(y))
    sse = 0
    for i in range(0, len(x)):
        sse += (x[i]-centroid[0])**2 + (y[i]-centroid[1])**2
    print("SSE: ", sse)

    if legend:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig, ax = plt.subplots(figsize=(12, 6))

    if D:
        fig = plt.figure(figsize=(12,6))
        ax = fig.gca(projection='3d', facecolor="white")

    if not D and not facet:
        data = pd.DataFrame(list(zip(x, y, labels_plot_colors)), columns=["x", "y", "Concept"])
        sns.set_palette(sns.color_palette("muted", len(labels_plot)))
        scatter = sns.scatterplot(data=data, x="x", y="y", hue="Concept", s=50)
        plt.ylim(-140, 140)
        plt.xlim(-140, 155)
    elif not facet:
        data = pd.DataFrame(list(zip(x, y, z, labels_value)), columns=["x", "y", "z", "Concept"])
        sns.set_palette(sns.color_palette("muted", len(labels_plot)))
        ax.scatter(data["x"], data["y"], data["z"], c=data["Concept"], cmap="tab10", marker='o', edgecolors="w")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.xlim(-35, 45)
        plt.ylim(-35, 30)
        ax.set_zlim(-35, 30)
    else:
        data = pd.DataFrame(list(zip(x, y, labels_plot_colors)), columns=["x", "y", "Concept"])
        g = sns.FacetGrid(data, col="Concept", hue="Concept")
        g = (g.map(plt.scatter, "x", "y", edgecolor="w"))


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
        if not D:
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
    parser.add_argument("--D", action="store_true", help="Assign label randomly", default=False)

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
    tnse_plot(objects[0], emb, random_=args.random, save=args.save,
    legend=args.no_legend, elmo=args.elmo, D=args.D, facet=args.facetgrid)
