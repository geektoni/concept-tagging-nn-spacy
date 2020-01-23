# Dataset

This directory contains the dataset used for the project.
They are stored using **Git-LFS** and therefore they must be downloaded
separately from the code. The entire dataset occupies at least **600 MB**
on disk.

You need to install Git-LFS to download the data. Please follow the
official instructions for your platform at this link [here](https://git-lfs.github.com/).
Then, run the following commands in your terminal:

```bash
cd concept-tagging-nn-spacy
git lfs pull
```

You can also run the following alternative set of instructions:
```bash
cd concept-tagging-nn-spacy
git lfs fetch
git lfs checkout
```

## Description

There are the following data in this directory. They are all compressed using
are generate using the `pickle` utility and they are compressed with the bzip2 format:
* **train.bz2 / test.bz2**: contains the original data (tokens, NER, POS, etc.) without any modifications;
* **train_bert.bz2 / test_bert.bz2**: updated version of the original dataset which includes a `tok_emb` column
which stores the BERT embeddings for each token;
* **train_elmo.bz2 / test_elmo.bz2**: update version of the original dataset which includes the ELMo embeddings for each
of the tokens.

The directory `embeddings` contains the file which has the ConceptNet embedding matrix.
It is saved with the bzip2 format.

There are also a couple of utility scripts:
* `convert_conceptnet_embeddings.py`: convert the original ConceptNet embeddings into the format (pickle,bz2) needed by
the models.
* `convert_glove_embeddings.py`: convert the original GloVe embeddings in the format needed by the project
(in the end the GloVe embeddings were abandoned in favour of ConceptNet).


