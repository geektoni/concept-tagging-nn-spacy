from bert_embedding import BertEmbedding

class BertEmbedderTransformer:

    def __init__(self):
        pass

    def __call__(self, data):
        bert_embedding = BertEmbedding(model="bert_12_768_12")
        emb = bert_embedding(data, 'sum')
        return emb