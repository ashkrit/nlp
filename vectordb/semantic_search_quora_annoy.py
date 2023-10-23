from sentence_transformers import SentenceTransformer, util
import os
import csv
import pickle
import time
import torch
from annoy import AnnoyIndex

model_name = 'quora-distilbert-multilingual'
model = SentenceTransformer(model_name)

url = "http://qim.fs.quoracdn.net/quora_duplicate_questions.tsv"
dataset_path = "quora_duplicate_questions.tsv"
max_corpus_size = 10000

n_trees = 256           #Number of trees used for Annoy. More trees => better recall, worse run-time
embedding_size = 768    #Size of embeddings
top_k_hits = 10         #Output k hits

annoy_index_path = 'quora-embeddings-{}-size-{}-annoy_index-trees-{}.ann'.format(model_name.replace('/', '_'), max_corpus_size,n_trees)
embedding_cache_path = 'quora-embeddings-{}-size-{}.pkl'.format(model_name.replace('/', '_'), max_corpus_size)

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

if not os.path.exists(annoy_index_path):
    # Create Annoy Index
    print("Create Annoy index with {} trees. This can take some time.".format(n_trees))
    annoy_index = AnnoyIndex(embedding_size, 'angular')

    for i in range(len(corpus_embeddings)):
        annoy_index.add_item(i, corpus_embeddings[i])

    annoy_index.build(n_trees)
    annoy_index.save(annoy_index_path)
else:
    #Load Annoy Index from disc
    annoy_index = AnnoyIndex(embedding_size, 'angular')
    annoy_index.load(annoy_index_path)


corpus_embeddings = torch.from_numpy(corpus_embeddings)

######### Search in the index ###########

print("Corpus loaded with {} sentences / embeddings".format(len(corpus_sentences)))


while True:
    inp_question = input("Please enter a question: ")

    start_time = time.time()
    question_embedding = model.encode(inp_question)

    corpus_ids, scores = annoy_index.get_nns_by_vector(question_embedding, top_k_hits, include_distances=True)
    hits = []
    for id, score in zip(corpus_ids, scores):
        hits.append({'corpus_id': id, 'score': 1-((score**2) / 2)})

    end_time = time.time()

    print("Input question:", inp_question)
    print("Results (after {:.3f} seconds):".format(end_time-start_time))
    for hit in hits[0:top_k_hits]:
        print("\t{:.3f}\t{}".format(hit['score'], corpus_sentences[hit['corpus_id']]))

    # Approximate Nearest Neighbor (ANN) is not exact, it might miss entries with high cosine similarity
    # Here, we compute the recall of ANN compared to the exact results
    correct_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k_hits)[0]
    correct_hits_ids = set([hit['corpus_id'] for hit in correct_hits])

    #Compute recall
    ann_corpus_ids = set(corpus_ids)
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