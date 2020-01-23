#
# This software is distributed under MIT license (see LICENSE file).
#
# Authors: Giovanni De Toni
#
import argparse
import pandas as pd
import numpy as np

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.6)

selected_tokens = ["O", "movie.name", "director.name", "actor.name", "producer.name", "movie.language", "movie.release_date"]
least_tokens = ["movie.genre", "character.name", "movie.gross_revenue", "movie.location", "award.ceremony", "movie.release_region", "movie.type"]

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate confusion matrix for a set of concepts given the results')
    parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the result file")

    args = parser.parse_args()

    print("Reading results")
    total_results = []
    total_results_least = []
    with (open(args.filename[0], "r")) as openfile:
        for l in openfile:

            if l == "\n":
                continue

            tokens = l.split(" ")
            t1 = tokens[1].split("-")
            t2= tokens[2].replace("\n","").split("-")

            if t1[len(t1)-1] in selected_tokens or t2[len(t2)-1] in selected_tokens:
                total_results.append([t1[len(t1)-1], t2[len(t2)-1]])

            if t1[len(t1)-1] in least_tokens or t2[len(t2)-1] in least_tokens:
                total_results_least.append([t1[len(t1)-1], t2[len(t2)-1]])


    # convert results into pandas
    print("Generating dataframe")
    df = pd.DataFrame(total_results, columns=["predicted", "original"])
    df2 = pd.DataFrame(total_results_least, columns=["predicted", "original"])

    # Generate confusion matrix
    cm = confusion_matrix(df["original"], df["predicted"], labels=df["original"].unique())
    cm2 = confusion_matrix(df2["original"], df2["predicted"], labels=df2["original"].unique())

    # Print total accuracy
    print("Accuracy: ", np.trace(cm)/np.sum(cm))
    print(np.diagonal(cm) / np.sum(cm, axis=1))

    cm_df = pd.DataFrame(cm, columns=df["original"].unique(), index=df["original"].unique())
    cm_df2 = pd.DataFrame(cm2, columns=df2["original"].unique(), index=df2["original"].unique())


    print("Generating the heatmap")
    # Print the heatmap
    plt.figure(figsize=(12,12))
    sns.heatmap(cm_df, vmin=cm_df.values.min(), vmax=cm_df.values.max(), fmt="d", annot=True, cmap="YlGnBu", square=True,
                annot_kws={"fontsize": 22},cbar_kws={"shrink": 0.5})
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=200)

    plt.figure(figsize=(12, 12))
    sns.heatmap(cm_df2, vmin=cm_df2.values.min(), vmax=cm_df2.values.max(), fmt="d", annot=True, cmap="YlGnBu",
                square=True,
                annot_kws={"fontsize": 22}, cbar_kws={"shrink": 0.5})
    plt.tight_layout()
    plt.savefig("correlation_matrix_least.png", dpi=200)

