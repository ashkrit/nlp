"""
This example computes the score between a query and all possible
sentences in a corpus using a Cross-Encoder for semantic textual similarity (STS).
It output then the most similar sentences for the given query.
"""
from sentence_transformers.cross_encoder import CrossEncoder
import numpy as np

# Pre-trained cross encoder
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# We want to compute the similarity between the query sentence
query =  "select * from transactions where subcategory=kids and bank=bank89 "

# With all sentences in the corpus
corpus = [
    "select * from transactions where bank=bank1 and merchant_country=USA.",
    "select * from transactions where bank=bank1 and merchant_country=SG.",
    "select * from transactions where bank=bank2 and channel=instore and card=debit.",
    "select * from transactions where bank=bank2 and channel=ecom and card=debit.",
    "select * from transactions where bank=bank2 and merchant=bill",
    "select * from transactions where bank=bank5 and category=Books",
    "select * from transactions where category=Books and bank=bank5",
    "select * from transactions where category=Books and bank=bank5 and subcategory=kids",
]

# So we create the respective sentence combinations
sentence_combinations = [[query, corpus_sentence] for corpus_sentence in corpus]

# Compute the similarity scores for these combinations
similarity_scores = model.predict(sentence_combinations)

# Sort the scores in decreasing order
sim_scores_argsort = reversed(np.argsort(similarity_scores))

# Print the scores
print("Query:", query)
for idx in sim_scores_argsort:
    print("{:.2f}\t{}".format(similarity_scores[idx], corpus[idx]))