#
# This software is distributed under MIT license (see LICENSE file).
#
# Authors: Giovanni De Toni
#

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import spacy

# https://courses.washington.edu/hypertxt/csar-v02/penntable.html
tagdict = [
"CC",
"CD",
"DT",
"EX",
"FW",
"IN",
"IN/that",
"JJ",
"JJR",
"JJS",
"LS",
"MD",
"NN",
"NNS",
"NP",
"NPS",
"PDT",
"POS",
"PP",
"PP$",
"RB",
"RBR",
"RBS",
"RP",
"SENT",
"SYM",
"TO",
"UH",
"VB",
"VBD",
"VBG",
"VBN",
"VBZ",
"VBP",
"VD",
"VDD",
"VDG",
"VDN",
"VDZ",
"VDP",
"VH",
"VHD",
"VHG",
"VHN",
"VHZ",
"VHP",
"VV",
"VVD",
"VVG",
"VVN",
"VVP",
"VVZ",
"WDT",
"WP",
"WP$",
"WRB",
":",
"$"]

# NER entities of spaCy
nerdict = ["PERSON", "NORP", "FAC", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "LAW",
           "LANGUAGE", "DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def get_wordnet_pos(treebank_tag):
    """
    Convert the POS tag into an wordnet pos tag.
    :param treebank_tag: treebank_tag
    :return: the converted tag
    """

    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ""

def get_lemmatize_word(word):
    """
    Lemmatize the given word
    :param word: string
    :return: lemmatized word
    """
    token = word_tokenize(word)
    pos_wn = nltk.pos_tag(token)[0][1]
    pos = "n" if get_wordnet_pos(pos_wn) == "" else get_wordnet_pos(pos_wn)
    return lemmatizer.lemmatize(word, pos=pos)

def ie_preprocess(phrase):
    """
    Tokenize and POS-tag the given phrase
    :param phrase: a string
    :return: the pos-tagged version of the phrase.
    """
    return nltk.pos_tag(nltk.word_tokenize(phrase))


def get_combined_representation(phrase, lemmas, pos, concepts, ner=[]):
    """
    Generate a combined representation of the phrase,lemmas,pos,concepts and ner
    :param phrase: current phrase
    :param lemmas: current lemmas
    :param pos: POS tags for the phrase
    :param concepts: concepts for the phrase
    :param ner: NER recognition for the phrase
    :return: a concatenation of the previous elements as a list
    """
    result = []
    if len(ner) != 0:
        for v in zip(phrase, lemmas, pos, concepts, ner):
            result.append(v[0]+v[1]+v[2]+v[3]+v[4])
    else:
        for v in zip(phrase, lemmas, pos, concepts):
            result.append(v[0]+v[1]+v[2]+v[3])
    return result


def return_replaced_concepts(tokens, concepts, method="keep"):
    """
    Replace a list of tokens in which they are replaced by using
    a certain method.
    :param tokens: the tokens we want to replace
    :param concepts: the current concepts associated to the tokens
    :param method: replacement method (keep, lemma, stem, word)
    :return: a list of replaced tokens
    """
    if method=="keep":
        return concepts
    elif method=="lemma":
        return [get_lemmatize_word(tokens[i]) if concepts[i] == "O" else concepts[i] for i in range(0, len(tokens))]
    elif method=="stem":
        return [stemmer.stem(tokens[i]) if concepts[i] == "O" else concepts[i] for i in range(0, len(tokens))]
    elif method=="word":
        return [tokens[i] if concepts[i] == "O" else concepts[i] for i in range(0, len(tokens))]


def ner_tool(row, method="spacy", replace_O="keep"):
    """
    Replace entities detected into the given phrase with
    an entity definition
    :param phrase: a string
    :param concepts: the concepts associated to the tokens
    :param pos: the pos associated to the tokens
    :param method: the ER method (spacy/nltk/none)
    :return: 'tokens', 'lemmas', 'pos', 'concepts', 'combined'
    """

    phrase, lemmas, pos, concepts = row["tokens"].copy(), row["lemmas"], row["pos"], row["concepts"]

    if method=="spacy":
        doc = nlp(" ".join(phrase))

        for entity in doc.ents:

            words = entity.text.split(" ")
            label = entity.label_

            indexes = []
            try:
                for w in words:
                    indexes.append(phrase.index(w))
                for i in range(0, len(indexes)):
                    phrase[indexes[i]] = "_{}".format(label.lower())
            except:
                # there are errors? Then we do not include that
                # phrase in the final dataset
                break

    return row["tokens"], row["lemmas"], row["pos"], row["concepts"], phrase, \
            get_combined_representation(row["tokens"], row["lemmas"], row["pos"], row["concepts"], phrase)

def one_hot_encoding_ner(ner):
    """
    Generate an one-hot encoding of the given NER entities
    :param ner: a list with the NER entities
    :return: a matrix with all the one-hot encoding
    """

    total_enc = []
    for e in ner:
        enc = [0 for i in range(0, len(nerdict))]
        if e.startswith("_"):
            try:
                index = nerdict.index(e.replace("_", "").upper())
            except:
                index = -1

            if index != -1:
                enc[index] = 1

        total_enc.append(enc)
    return total_enc


def one_hot_encoding_pos(pos):
    """
    Generate an one-hot-encoding representation for the Part Of Speech values.
    :param pos: list with all the POS tags
    :return: a list with the one-hot vector conversion.
    """

    total_enc = []
    for ptag in pos:
        enc = [0 for i in range(len(tagdict))]

        # If we cannot find the tag then we use an
        # empty vector
        try:
            index = tagdict.index(ptag)
        except:
            index = -1

        if index != -1:
            enc[index] = 1

        total_enc.append(enc)
    return total_enc