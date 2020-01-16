import torch
from allennlp.commands.elmo import ElmoEmbedder

class ElmoEmbedderTransformer():

    def __init__(self):
        self.elmo = ElmoEmbedder()

    def __call__(self, data):
        result = self.elmo.embed_sentence(data)
        complete_embeddings = 1/3 * result[0] + 1/3 * result[1] + 1/3 * result[2]
        return complete_embeddings