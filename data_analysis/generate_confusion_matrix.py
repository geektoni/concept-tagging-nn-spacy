import argparse
import pandas as pd

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(font_scale=1.6)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inspect pickle files used for training/testing')
    parser.add_argument("filename", metavar='file', nargs=1, type=str, help="Path to the file we want to explore")

    args = parser.parse_args()

    print("Reading results")
    total_results = []
    with (open(args.filename[0], "r")) as openfile:
        for l in openfile:

            if l == "\n":
                continue

            tokens = l.split(" ")
            t1 = tokens[1].split("-")
            t2= tokens[2].replace("\n","").split("-")
            total_results.append([t1[len(t1)-1], t2[len(t2)-1]])


    # convert results into pandas
    print("Generating dataframe")
    df = pd.DataFrame(total_results, columns=["predicted", "original"])

    # Generate confusion matrix
    cm = confusion_matrix(df["original"], df["predicted"], labels=df["original"].unique())

    cm_df = pd.DataFrame(cm, columns=df["original"].unique(), index=df["original"].unique())

    print("Generating the heatmap")
    # Print the heatmap
    plt.figure(figsize=(12,12))
    sns.heatmap(cm_df, vmin=cm_df.values.min(), vmax=cm_df.values.max(), fmt="d", annot=True, cmap="YlGnBu", square=True,
                annot_kws={"fontsize": 12},cbar_kws={"shrink": 0.5})
    plt.tight_layout()
    plt.savefig("correlation_matrix.png", dpi=200)

