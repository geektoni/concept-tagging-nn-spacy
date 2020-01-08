import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")
lemmatizer = WordNetLemmatizer()

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

def get_lemmatize_word(phrase):
    """
    Lemmatize the given word
    :param word: string
    :return: lemmatized word
    """
    result = []
    for word in phrase:
        token = word_tokenize(word)
        pos_wn = nltk.pos_tag(token)[0][1]
        pos = "n" if get_wordnet_pos(pos_wn) == "" else get_wordnet_pos(pos_wn)
        result.append(lemmatizer.lemmatize(word, pos=pos))
    return result

def ie_preprocess(phrase):
    """
    Tokenize and POS-tag the given phrase
    :param phrase: a string
    :return: the pos-tagged version of the phrase.
    """
    return nltk.pos_tag(nltk.word_tokenize(phrase))


def get_combined_representation(phrase, lemmas, pos, concepts):
    result = []
    for v in zip(phrase, lemmas, pos, concepts):
        result.append(v[0]+v[1]+v[2]+v[3])
    return result


def ner_tool(row, method="none"):
    """
    Replace entities detected into the given phrase with
    an entity definition
    :param phrase: a string
    :param concepts: the concepts associated to the tokens
    :param pos: the pos associated to the tokens
    :param method: the ER method (spacy/nltk/none)
    :return: 'tokens', 'lemmas', 'pos', 'concepts', 'combined'
    """

    phrase, lemmas, pos, concepts = row["tokens"], row["lemmas"], row["pos"], row["concepts"]

    if method=="none":
        return phrase, lemmas, pos, concepts, row["combined"]

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
    else:
        sent = ie_preprocess(" ".join(phrase))
        result = nltk.ne_chunk(sent)
        result_bin = nltk.ne_chunk(sent, binary=True)
        for i in range(0, len(result_bin)):
            if result_bin[i][1] == "NE":
                phrase[i] = "_{}".format(result[i][1].lower())

    # Remove duplicate occurrences
    i = 0
    while i < len(phrase) - 1:
        if phrase[i] == phrase[i + 1] and phrase[i].startswith("_"):
            del phrase[i]
            del concepts[i]
            del lemmas[i]
            del pos[i]
        else:
            i = i + 1

    return phrase, lemmas, pos, concepts, get_combined_representation(phrase, lemmas, pos, concepts)