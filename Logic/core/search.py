import json
import numpy as np
from .preprocess import Preprocessor
from .scorer import Scorer
from .indexer.indexes_enum import Indexes, Index_types
from .indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        # path = './index/'
        path = 'C:/Users/ALIREZA/Desktop/IMDB-IR-System/Logic/core/indexer/index/'
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

    def search(self, query, method, weights, safe_ranking = True, max_results=10):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results. 
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split()

        scores = {}
        if safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

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
    method = "lnc.ltc"
    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights)

    print(result)
