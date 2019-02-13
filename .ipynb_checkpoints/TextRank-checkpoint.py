from nltk.corpus import reuters
import networkx as nx
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize, sent_tokenize
from numpy import dot
import operator
from numpy.linalg import norm

def summarize(article, articles, do_train, num_sents=5):

    model = 0
    
    if do_train:
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(articles)]

        max_epochs = 10
        vec_size = 20
        alpha = 0.025

        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha, 
                        min_alpha=0.00025,
                        min_count=1,
                        dm =1)

        model.build_vocab(tagged_data)

        for epoch in range(max_epochs):
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

            model.save("d2v.model")

    else:
        model = Doc2Vec.load("d2v.model")

    
    G = nx.Graph()
    article_tok = sent_tokenize(article)

    for sentence in article_tok:
        for inner in article_tok:
            if not sentence == inner:
                test_data1 = word_tokenize(sentence.lower())
                test_data2 = word_tokenize(inner.lower())
                v1 = model.infer_vector(test_data1)
                v2 = model.infer_vector(test_data2)
                cos_sim = dot(v1, v2)/(norm(v1)*norm(v2))
                G.add_edge(sentence, inner, weight=cos_sim)

    ranked = []
    i = 0
    for n, nbrs in G.adj.items():
        total = 0
        for nbr, eattr in nbrs.items():
            wt = nbrs[nbr]["weight"]
            total += wt
        ranked.append((i, n, total))
        i += 1

    top_sents = list(reversed(sorted(ranked, key=operator.itemgetter(2))))[:num_sents]
    return [x[1] for x in list(sorted(top_sents, key=operator.itemgetter(0)))]