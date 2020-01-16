from allennlp.commands.elmo import ElmoEmbedder

class ElmoEmbedderTransformer():

    def __init__(self):
        self.elmo = ElmoEmbedder()

    def __call__(self, data):
        return self.elmo.embed_sentence(data)