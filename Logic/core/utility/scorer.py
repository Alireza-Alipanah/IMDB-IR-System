import numpy as np
from collections import Counter


class Scorer:
    def __init__(self, index, number_of_documents, b=0.75, k=1.6):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents
        self.b =b
        self.k = k
        self.dl = {}

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.

        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.

        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))

    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.

        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            if term not in self.index:
                return 0
            df = len(self.index[term].values())
            self.idf[term] = np.log(self.N / df)
        return self.idf[term]

    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """

        return dict(Counter(query))

    def apply_method(self, tf_dict, method):
        for key, value in tf_dict.items():
            if method[0] == 'l' and value != 0:
                    value = 1 + np.log(value)
            if method[1] == 'n':
                continue
            tf_dict[key] = value * self.get_idf(key)
        if method[2] == 'n':
            return tf_dict
        normalizing_factor = np.linalg.norm(list(tf_dict.values()))
        return {key: value / normalizing_factor for key, value in tf_dict.items()}

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        query_method, document_method = method.split('.')
        query_scores = self.get_query_tfs(query)
        query_scores = self.apply_method(query_scores, query_method)
        list_of_documents = self.get_list_of_documents(query)
        documents_scores = {}
        for document_id in list_of_documents:
            documents_scores[document_id] = self.get_vector_space_model_score(query, query_scores, document_id, document_method)
        return documents_scores

    def get_vector_space_model_score(
        self, query, query_scores, document_id, document_method
    ):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        documents_score = {}
        for term in query:
            documents_score[term] = self.index.get(term, {}).get(document_id, 0)
        documents_score = self.apply_method(documents_score, document_method)
        return sum(documents_score.get(term, 0) * query_scores[term] for term in query)

    def compute_socres_with_okapi_bm25(
        self, query, average_document_field_length, document_lengths
    ):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        list_of_documents = self.get_list_of_documents(query)
        return {document_id: self.get_okapi_bm25_score(query, document_id, average_document_field_length,
                                                        document_lengths) for document_id in list_of_documents}

    def get_okapi_bm25_score(
        self, query, document_id, average_document_field_length, document_lengths
    ):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        return sum(
            self.get_idf(term) * (self.k + 1) * self.index.get(term, {}).get(document_id, 0) / ( \
                self.k * ((1 - self.b) + self.b * document_lengths[document_id] \
                           / average_document_field_length) + self.index.get(term, {}).get(document_id, 0)
            )
            for term in query
        )

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, document_unique_lengths=None, collection_index=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """
        query_tfs = self.get_query_tfs(query)
        list_of_documents = self.get_list_of_documents(query)
        return {document_id: self.compute_score_with_unigram_model(query_tfs, document_id, smoothing_method,
                                                        document_lengths, document_unique_lengths, collection_index,
                                                        alpha, lamda) for document_id in list_of_documents}

    def get_naive_smoothing_probability(self, query_term, document_id, document_length, document_unique_length, k=1):
        return (self.index.get(query_term, {}).get(document_id, 0) + k) / \
            (document_length + document_unique_length * k)

    def get_bayes_smoothing_probability(self, query_term, document_id, document_length, collection_index, alpha):
        return (self.index.get(query_term, {}).get(document_length, 0) + alpha * collection_index.get(query_term, 0)) / \
            (document_length + alpha)

    def get_mixture_probability(self, query_term, document_id, document_length, collection_index, lamda):
        return lamda * self.index.get(query_term, {}).get(document_id, 0) / document_length + \
            (1 - lamda) * collection_index.get(query_term, 0)


    def compute_score_with_unigram_model(
        self, query_tfs, document_id, smoothing_method, document_lengths, document_unique_lengths, collection_index
        , alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """
        prob = 1
        doc_len = document_lengths[document_id]
        document_unique_length = document_unique_lengths[document_id]
        for query_term, count in query_tfs.items():
            if smoothing_method == 'naive':
                query_prob =  self.get_naive_smoothing_probability(query_term, document_id, doc_len, document_unique_length)
            elif smoothing_method == 'bayes':
                query_prob = self.get_bayes_smoothing_probability(query_term, document_id, doc_len, collection_index, alpha)
            elif smoothing_method == 'mixture':
                query_prob = self.get_mixture_probability(query_term, document_id, doc_len, collection_index, lamda)
            elif smoothing_method is None:
                query_prob = self.index.get(query_term, {}).get(document_id, 0) / doc_len
            else:
                raise Exception(f'{smoothing_method} smoothing method not implemented.')
            prob = prob * (query_prob ** count)
        return prob

