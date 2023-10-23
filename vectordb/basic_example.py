from sentence_transformers import SentenceTransformer
import logging


logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    model_name="all-MiniLM-L6-v2"
    model = SentenceTransformer(model_name)
    logging.info(model)


    #Our sentences we like to encode
    sentences = ['This framework generates embeddings for each input sentence',
    'Sentences are passed as a list of string.', 
    'The quick brown fox jumps over the lazy dog.']


    #Sentences are encoded by calling model.encode()
    sentence_embeddings = model.encode(sentences)

    #Print the embeddings
    for sentence, embedding in zip(sentences, sentence_embeddings):
        logging.info(f"Sentence: {sentence}")
        logging.info(f"Embedding: {embedding}")
        logging.info("")

   