import networkx as nx                                     # Graphs
from gensim.models.doc2vec import Doc2Vec, TaggedDocument # Sent2Vec functionality
from nltk.tokenize import word_tokenize, sent_tokenize    # Sentence and word tokenization
from numpy import dot                                     # Fast dot product
import operator                                           # Fast touple indexing
from numpy.linalg import norm                             # Fast vector normalization 

def summarize(article, articles, do_train=False, num_sents=5):
    """Summarize an article using TextRank

    Args:
        article: The document to summarize
        articles: The corpus to "train" on
        do_train: (False) if the model needs retrained on the corpus passed in
        num_sents: (5) Number of sentences to return as the summary

    Returns:
        Top n sentences that summarize the document

    """
    # Initialize the model as nothing
    model = None
    
    # If the train flag is set then train the model
    if do_train:
        # Tag the data
        tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(articles)]

        # Set the meta-variables
        max_epochs = 10
        vec_size = 20
        alpha = 0.025

        # Create the model (not trained)
        model = Doc2Vec(vector_size=vec_size,
                        alpha=alpha, 
                        min_alpha=0.00025,
                        min_count=1,
                        dm =1)

        # Build the models vocabulary
        model.build_vocab(tagged_data)

        # Train the model for each epoch
        for epoch in range(max_epochs):
            model.train(tagged_data,
                        total_examples=model.corpus_count,
                        epochs=model.epochs)
            # decrease the learning rate
            model.alpha -= 0.0002
            # fix the learning rate, no decay
            model.min_alpha = model.alpha

            # Save the trained model
            model.save("d2v.model")

    # If the train flag is not set
    else:
        # Load the model 
        model = Doc2Vec.load("d2v.model")

    # Create a new graph
    G = nx.Graph()
    
    # Split the article into sentences
    article_tok = sent_tokenize(article)

    # For each sentence
    for sentence in article_tok:
        # For each sentence again
        for inner in article_tok:
            # If the two sentences are not the same
            if not sentence == inner:
                # Get the vectors for each sentence
                test_data1 = word_tokenize(sentence.lower())
                test_data2 = word_tokenize(inner.lower())
                v1 = model.infer_vector(test_data1)
                v2 = model.infer_vector(test_data2)
                
                # Calculate the cosine similarity between the vectors
                cos_sim = dot(v1, v2)/(norm(v1)*norm(v2))
                
                # Add a new edge between the two nodes with the weight being the cosine similarity
                G.add_edge(sentence, inner, weight=cos_sim)

    # Create a new ranked array and counting variable
    ranked = []
    i = 0
    
    # Iterate through the graph
    for n, nbrs in G.adj.items():
        # Sum for the total number of neighbors
        total = 0
        for nbr, eattr in nbrs.items():
            # Append the weight to the total
            wt = nbrs[nbr]["weight"]
            total += wt
        # Append the order, sentence, and score to the array
        ranked.append((i, n, total))
        i += 1

    # Return the top n sentences sorted by the score and then order seen in the document
    top_sents = list(reversed(sorted(ranked, key=operator.itemgetter(2))))[:num_sents]
    return [x[1] for x in list(sorted(top_sents, key=operator.itemgetter(0)))]