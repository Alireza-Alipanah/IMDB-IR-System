import json
import numpy as np
import os

# from .indexer import Indexes, Index_types, Index_reader
try:
    from utility.preprocess import Preprocessor
    from utility.scorer import Scorer
    from indexer.indexes_enum import Indexes, Index_types
    from indexer.index_reader import Index_reader
except Exception:
    from .utility.preprocess import Preprocessor
    from .utility.scorer import Scorer
    from .indexer.indexes_enum import Indexes, Index_types
    from .indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'indexer', 'index')
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES).index
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED).index
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH).index
        }
        self.document_unique_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_UNIQUE_LENGTH).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_UNIQUE_LENGTH).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_UNIQUE_LENGTH).index
        }
        self.collection_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.COLLECTION).index,
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.COLLECTION).index,
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.COLLECTION).index
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA).index

        self.safe_scorers = {
            Indexes.STARS: Scorer(self.document_indexes[Indexes.STARS], self.metadata_index['document_count']),
            Indexes.GENRES: Scorer(self.document_indexes[Indexes.GENRES], self.metadata_index['document_count']),
            Indexes.SUMMARIES: Scorer(self.document_indexes[Indexes.SUMMARIES], self.metadata_index['document_count'])
        }
        self.unsafe_scorers = {}
        for tier in ["first_tier", "second_tier", "third_tier"]:
            self.unsafe_scorers[tier] = {
                Indexes.STARS: Scorer(self.tiered_index[Indexes.STARS][tier], self.metadata_index['document_count']),
                Indexes.GENRES: Scorer(self.tiered_index[Indexes.GENRES][tier], self.metadata_index['document_count']),
                Indexes.SUMMARIES: Scorer(self.tiered_index[Indexes.SUMMARIES][tier], self.metadata_index['document_count'])
            }


    def search(
        self,
        query,
        method,
        weights,
        safe_ranking=True,
        max_results=10,
        smoothing_method=None,
        alpha=0.5,
        lamda=0.5,
    ):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25 | Unigram
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results.
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """
        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split()

        scores = {}
        if method == "Unigram":
            self.find_scores_with_unigram_model(
                query, smoothing_method, weights, scores, alpha, lamda
            )
        elif safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(
                query, method, weights, max_results, scores
            )

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)
        
        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        for field, value in weights.items():
            if field not in scores:
                continue
            document_scores = scores[field]
            for document_id in document_scores.keys():
                if document_id not in final_scores:
                    final_scores[document_id] = 0
                final_scores[document_id] += value * document_scores[document_id]


    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for field, value in weights.items():
            prev_score = {}
            for tier in ["first_tier", "second_tier", "third_tier"]:
                if value == 0:
                    continue
                if method != 'OkapiBM25':
                    score = self.unsafe_scorers[tier][field].compute_scores_with_vector_space_model(query, method)
                else:
                    score = self.unsafe_scorers[tier][field].compute_socres_with_okapi_bm25(query, \
                                        self.metadata_index[field.value], self.document_lengths_index(field.value))
                prev_score = self.merge_scores(score, prev_score)
                if len(prev_score.keys()) >= max_results:
                    break
            scores[field] = prev_score

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """
        for field, value in weights.items():
            if value == 0:
                continue
            if method != 'OkapiBM25':
                scores[field] = self.safe_scorers[field].compute_scores_with_vector_space_model(query, method)
            else:
                scores[field] = self.safe_scorers[field].compute_socres_with_okapi_bm25(query,
                                    self.metadata_index['averge_document_length'][field.value],
                                      self.document_lengths_index[field])

                                      
    def find_scores_with_unigram_model(
        self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        weights : dict
            A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
        scores : dict
            The scores of the documents.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.
        """
        for field, value in weights.items():
            if value == 0:
                continue
            scores[field] = self.safe_scorers[field].compute_scores_with_unigram_model(
                query,
                smoothing_method,
                self.document_lengths_index[field],
                self.document_unique_lengths_index[field],
                self.collection_index[field],
                alpha,
                lamda
            )


    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """
        for key, value in scores2:
            if key not in scores1:
                scores1[key] = value
            else:
                scores1[key] += value
        return scores1


if __name__ == '__main__':
    search_engine = SearchEngine()
    query = "spider man in wonderland"
    method = "Unigram"
    smoothing_method = 'mixture'
    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights, smoothing_method=smoothing_method)

    print(result)
