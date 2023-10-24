from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import faiss
import numpy as np
import math



model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)

url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 10000

embedding_cache_path = 'quora-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)


embedding_size = 768    #Size of embeddings
top_k_hits = 10         #Output k hits


def create_flat_index() -> faiss.IndexFlat:
    return faiss.IndexFlatL2(embedding_size)

def create_flat_index_ip() -> faiss.IndexFlatIP:
    return faiss.IndexFlatIP(embedding_size)

def create_iv_index() -> faiss.IndexIVFFlat:
   quantizer = faiss.IndexFlatL2(embedding_size) 
   n_list = 100
   return faiss.IndexIVFFlat(quantizer, embedding_size, n_list)


def create_iv_pq_index() -> faiss.IndexIVFPQ:
   quantizer = faiss.IndexFlatL2(embedding_size) 
   m = 8
   n_clusters = 4
   return faiss.IndexIVFPQ(quantizer, embedding_size, n_clusters, m, 8)

index = create_iv_index()


#Check if embedding cache path exists
if not os.path.exists(embedding_cache_path):
    # Check if the dataset exists. If not, download and extract
    # Download dataset if needed
    if not os.path.exists(dataset_path):
        print("Download dataset")
        util.http_get(url, dataset_path)

    # Get all unique sentences from the file
    corpus_sentences = set()
    with open(dataset_path, encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_MINIMAL)
        for row in reader:
            corpus_sentences.add(row['question1'])
            if len(corpus_sentences) >= max_corpus_size:
                break

            corpus_sentences.add(row['question2'])
            if len(corpus_sentences) >= max_corpus_size:
                break

    corpus_sentences = list(corpus_sentences)
    print("Encode the corpus. This might take a while")
    corpus_embeddings = model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)

    print("Store file on disc")
    with open(embedding_cache_path, "wb") as fOut:
        pickle.dump({'sentences': corpus_sentences, 'embeddings': corpus_embeddings}, fOut)
else:
    print("Load pre-computed embeddings from disc")
    with open(embedding_cache_path, "rb") as fIn:
        cache_data = pickle.load(fIn)
        corpus_sentences = cache_data['sentences']
        corpus_embeddings = cache_data['embeddings']

### Create the FAISS index
print("Start creating FAISS index")
# First, we need to normalize vectors to unit length
corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]



#Number of clusters to explorer at search time. We will search for nearest neighbors in 3 clusters.
#index.nprobe = 3

print("Start training the FAISS index")
print(f"is trained {index.is_trained}")

if index.is_trained==False:
    # Then we train the index to find a suitable clustering
    print("Starting training")
    index.train(corpus_embeddings)

print("Start adding embeddings to the FAISS index")
# Finally we add all embeddings to the index
index.add(corpus_embeddings)


######### Search in the index ###########


print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))

while True:
    inp_question = input("Please enter a question: ")

    start_time = time.time()
    question_embedding = model.encode(inp_question)

    #FAISS works with inner product (dot product). When we normalize vectors to unit length, inner product is equal to cosine similarity
    question_embedding = question_embedding / np.linalg.norm(question_embedding)
    question_embedding = np.expand_dims(question_embedding, axis=0)

    # Search in FAISS. It returns a matrix with distances and corpus ids.
    distances, corpus_ids = index.search(question_embedding, top_k_hits)

    # We extract corpus ids and scores for the first query
    hits = [{'corpus_id': id, 'score': score} for id, score in zip(corpus_ids[0], distances[0])]
    hits = sorted(hits, key=lambda x: x['score'], reverse=True)
    end_time = time.time()

    print("Input question:", inp_question)
    print("Results (after {:.3f} seconds):".format(end_time-start_time))
    for hit in hits[0:top_k_hits]:
        print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

    # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
    # Here, we compute the recall of ANN compared to the exact results
    correct_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k_hits)[0]
    correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])

    ann_corpus_ids = set([hit['corpus_id'] for hit in hits])
    if len(ann_corpus_ids) != len(correct_hits_ids):
        print("Approximate Nearest Neighbor returned a different number of results than expected")

    recall = len(ann_corpus_ids.intersection(correct_hits_ids)) / len(correct_hits_ids)
    print("\nApproximate Nearest Neighbor Recall@{}: {:.2f}".format(top_k_hits, recall * 100))

    if recall < 1:
        print("Missing results:")
        for hit in correct_hits[0:top_k_hits]:
            if hit['corpus_id'] not in ann_corpus_ids:
                print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))
    print("\n\n========\n")