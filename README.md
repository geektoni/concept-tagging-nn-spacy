# Concept Tagging Sequences using Transfer Learning and Named Entity Recognition Tools

Neural Networks have produced remarkable results in many Natural Language Processing tasks, for example, when tasked 
to assigning concepts to words of a sentence. 
Their successes are made possible by employing good word representations (embeddings) which a Neural Network can understand. 
This work evaluates several newly developed pre-trained embeddings (ELMo, BERT and ConceptNet) on the task of tagging sequences from the movie domain. We then compare the measurements with previous results of the literature.

This repository contains the code for the second assignment of the Language Understanding Systems
course of University of Trento teached by professor Giuseppe Riccardi.

The final report can be found at this link [here](report/giovanni_de_toni_197814.pdf).

## Description

This repository is structured as follow:
* `concept-tagging-with-neural-networks`: this directory contains the code of the original work
on which we based the project. This is loaded as a submodule;
* `data`: this directory contains the datasets/embeddings used for the project. They are saved with
Git-LFS in a compressed format;
* `data_analysis`: this scripts contains utility to analyze the various datasets;
* `report`: it contains the report;

The scripts you can find here:
* `collect_results.sh`: collect the results and generate a complete file;
* `generate_result_table.py`: generate a table from the collected results;
* `train_all_models.py`: script to run the experiments of a HPC cluster. It produces
a series of jobs and the results will be saved in the directory `results`;
* `submit_jobs.sh`: submit a job inside the cluster;
* `train_models.sh`: script which runs exactly one run of the models.

## Install

This project was written using **Python 3.7**, **bash** and **conda** for
the virtual environment. Everything was tested on **Ubuntu 18.04** To install all the dependencies needed please run the
following commands.

```bash
git clone https://github.com/geektoni/concept-tagging-nn-spacy
cd concept-tagging-nn-spacy
git submodule update --init
conda env create -f environment.yml
python -m spacy download en_core_web_sm
conda activate ctnns
```
The next step will be downloading the datasets. You will need to install
**Git-LFS** to be able to do so. Please refer to the official instructions. Once installed,
please run the following commands.
```bash
cd concept-tagging-nn-spacy
git lfs fetch
git lfs checkout
```
If everything worked correctly, you should be able to have a working environment
were to run the various scripts/experimens.

If you encounter any errors, please feel free to open an issue on Github.

## Usage

In order to replicate the experiments, you can run the following commands:
