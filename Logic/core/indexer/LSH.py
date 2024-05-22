import json
import numpy as np
import itertools
import random
from functools import reduce


def generate_hash_function():
    a = random.randint(1, int(1e8))
    b = random.randint(1, int(1e8))
    return lambda x: (a * x + b) % (2**32)


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes

    def shingle_document(self, document, k=2):
        """
         Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        shingles = set(document[i:i+k] for i in range(len(document) - k + 1))
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        doc_shingles = [self.shingle_document(i) for i in self.documents]
        all_shingles = reduce(set.union, doc_shingles)
        characteristic_matrix = np.zeros((len(all_shingles), len(self.documents)))
        for i, shingle in enumerate(all_shingles):
            for j, shingles in enumerate(doc_shingles):
                if shingle in shingles:
                    characteristic_matrix[i][j] = 1
        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        hash_functions = [generate_hash_function() for _ in range(self.num_hashes)]
        characteristic_matrix = self.build_characteristic_matrix()
        signatures = np.zeros((self.num_hashes, characteristic_matrix.shape[1]))
        for i, hash_function in enumerate(hash_functions):
            for j in range(signatures.shape[1]):
                signature = min(hash_function(k) for k in range(characteristic_matrix.shape[0]) if characteristic_matrix[k][j] == 1)
                signatures[i][j] = signature
        return signatures


    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = {}
        for band_idx in range(bands):
            idx = band_idx * rows_per_band
            band_signatures = signature[idx: idx + rows_per_band, :]
            for col in range(band_signatures.shape[1]):
                hash_value = hash(tuple(band_signatures[:, col]))
                bucket_id = f"{band_idx}_{hash_value}"
                if bucket_id not in buckets:
                    buckets[bucket_id] = []
                buckets[bucket_id].append(col)
        return buckets



    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        signatures = self.min_hash_signature()
        buckets = self.lsh_buckets(signatures)
        return buckets

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        intersection = first_set.intersection(second_set)
        union = first_set.union(second_set)
        jaccard_score = len(intersection) / len(union)
        return jaccard_score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)


# code to the this file:
if __name__ == '__main__':
    print('reading files...')
    with open('LSHFakeData.json', 'r') as file:
        data_fake = json.load(file)
    # with open('IMDB_crawled.json', 'r') as file:
    #     data_crawled = json.load(file)
    print('getting document summaries...')
    documents = [' '.join(document['summaries']) for document in data_fake] 
                # [' '.join(document['summaries']) for document in data_crawled if document['summaries'] is not None]
    print('performin minhashlsh...')
    minhashlsh = MinHashLSH(documents, 128)
    buckets = minhashlsh.perform_lsh()
    print('testing...')
    minhashlsh.jaccard_similarity_test(buckets, documents)