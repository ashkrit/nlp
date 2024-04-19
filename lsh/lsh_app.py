from collections import Counter
from math import sqrt
from random import choices
from typing import Iterable, List, TypeVar

import numpy as np


vector = TypeVar('vector', List[int], List[float])


class SimHash:

    def __init__(self, hash_size: int, inp_dimensions: int):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = {}
        # simplified distribution, see wiki for details: https://en.wikipedia.org/wiki/Random_projection#cite_ref-4
        self.projections = np.array(
            [
                [
                    sqrt(3) * choices([1, 0, -1], [1/6, 2/3, 1/6])[0] for _ in range(self.inp_dimensions)
                ] for _ in range(self.hash_size)
            ]
        )

    def generate_hash(self, inp_vector: vector):
        binary_vector = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(binary_vector.astype('str'))

    def __setitem__(self, inp_vec: vector, label: str):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table.get(hash_value, []) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


class LSH:

    def __init__(self, buckets_number: int, hash_size: int, inp_dimensions: int):
        self.buckets_number = buckets_number
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = []
        for i in range(self.buckets_number):
            self.hash_tables.append(SimHash(self.hash_size, self.inp_dimensions))

    def __setitem__(self, inp_vec: vector, label: str):
        for table in self.hash_tables:
            table[inp_vec] = label

    def __getitem__(self, inp_vec: vector):
        results = []
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        buckets_counter = Counter(results)
        return [(result, buckets_counter.get(result)) for result in set(results)]


def cosine_similarity(vec1: vector, vec2: vector):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def words_to_vectors(words_map: dict, string: str) -> List[int]:
    sentence_vector = []
    sentence_partial = string.split()
    for key, index in words_map.items():
        if key in sentence_partial:
            sentence_vector.insert(index, 1)
        else:
            sentence_vector.insert(index, 0)
    return sentence_vector


def generate_set_of_words(strings: Iterable[str]) -> set:
    words_set = set()
    for string in strings:
        for word in string.split():
            words_set.add(word)
    return words_set


def generate_map_of_words(words: Iterable[str]) -> dict:
    words_mapping = {}
    counter = 1
    for word in words:
        words_mapping[word] = counter
        counter += 1
    return words_mapping


if __name__ == '__main__':
    threshold = 0.5
    buckets_number = 64
    hash_size = 9

    documents = [
        'brown fox jumps over the lazy dog',
        'brown frog jumps over the busy dog',
        'yellow parrot flee over the lazy dog',
        'someone was knocking at the door',
        'the dog was barking',
        'the cat was smiling'
    ]

    # extract unique words from all documents
    set_of_words = generate_set_of_words(documents)

    # construct the map of words in order to vectorize documents
    # words order won't be preserved on each new script run
    # but that's doesn't matter as long as order is preserved
    # during the run
    map_of_words = generate_map_of_words(set_of_words)

    # simply the number of unique words in all documents
    num_of_feature_dimensions = len(set_of_words)

    # here be dragons: all LSH magic happens inside two classes
    hash_table = LSH(buckets_number, hash_size, num_of_feature_dimensions)

    # vectorize words and add them to LSH
    for document in documents:
        vector = words_to_vectors(map_of_words, document)
        hash_table[vector] = document

    # new document we'd like to get ANN for
    new_doc = 'yellow cat jumps over the lazy dog'
    # convert it vector representation
    vector = words_to_vectors(map_of_words, new_doc)

    # find all occurrence for specified vector
    lsh_buckets = hash_table[vector]

    print(f'Original sentence: {new_doc}')
    for doc, buckets_count in lsh_buckets:
        # calculate cosine distance between specified and found vectors
        dist = cosine_similarity(vector, words_to_vectors(map_of_words, doc))
        # if distance is above threshold - show us the result
        if dist > threshold:
            print(
                f'Found duplicate with cosine dist of {dist} and buckets count of {buckets_count}: {doc}'
            )
    print('\n')