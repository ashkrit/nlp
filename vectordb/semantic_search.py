from sentence_transformers import SentenceTransformer,util
import logging


logging.basicConfig(level=logging.INFO)

if __name__ =="__main__":
    model_name="multi-qa-MiniLM-L6-cos-v1"
    model = SentenceTransformer(model_name)
    logging.info(model)

    query_embedding = model.encode('How big is London')
    passage_embedding = model.encode(['London has 9,787,426 inhabitants at the 2011 census',
                                  'London is known for its finacial district'])
    scores = util.dot_score(query_embedding, passage_embedding)
    logging.info(f"Similarity: {scores}")
