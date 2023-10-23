<<<<<<< HEAD
from sentence_transformers import SentenceTransformer, util
import logging
import torch
=======
from sentence_transformers import SentenceTransformer,util
import logging
>>>>>>> a700d6fd0c0460821a647ada5f2f4e8421cbb17f


logging.basicConfig(level=logging.INFO)

<<<<<<< HEAD
if __name__ == "__main__":
    model_name = "multi-qa-MiniLM-L6-cos-v1"
=======
if __name__ =="__main__":
    model_name="multi-qa-MiniLM-L6-cos-v1"
>>>>>>> a700d6fd0c0460821a647ada5f2f4e8421cbb17f
    model = SentenceTransformer(model_name)
    logging.info(model)

    query_embedding = model.encode('How big is London')
    passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
<<<<<<< HEAD
                                      'London is known for its finacial district'])
    scores = util.dot_score(query_embedding, passage_embedding)
    logging.info(f"Similarity: {scores}")

    logging.info("Another example ")

    model_name = "all-MiniLM-L6-v2"
    embedder = SentenceTransformer(model_name)

    # Corpus with example sentences
    corpus = ['A man is eating food.',
              'A man is eating a piece of bread.',
              'The girl is carrying a baby.',
              'A man is riding a horse.',
              'A woman is playing violin.',
              'Two men pushed carts through the woods.',
              'A man is riding a white horse on an enclosed ground.',
              'A monkey is playing drums.',
              'A cheetah is running behind its prey.'
              ]
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    # Query sentences:
    queries = ['A man is eating pasta.', 'Someone in a gorilla costume is playing a set of drums.', 'A cheetah chases prey on across a field.']

    top_k = min(5, len(corpus))

    for query in queries:
        query_embedding = embedder.encode(query, convert_to_tensor=True)
        cosin_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cosin_scores, k=top_k)

        
        logging.info("====================")
        logging.info(top_results)
        logging.info(f"Query : {query}")
        logging.info(f"Matching result for {top_k} rows")

        for score , index in zip(top_results[0], top_results[1]):
            logging.info(f"{corpus[index]} \t {score}")

    
=======
                                  'London is known for its finacial district'])
    scores = util.dot_score(query_embedding, passage_embedding)
    logging.info(f"Similarity: {scores}")
>>>>>>> a700d6fd0c0460821a647ada5f2f4e8421cbb17f
