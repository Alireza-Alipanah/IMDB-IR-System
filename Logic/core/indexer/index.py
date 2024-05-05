import time
import os
import json
import copy
try:
    from .indexes_enum import Indexes
except ImportError:
    from indexes_enum import Indexes


class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        current_index = {preprocessed_document['id']: preprocessed_document for preprocessed_document in self.preprocessed_documents}
        return current_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        star_index = {}
        for document in self.preprocessed_documents:
            if 'stars' not in document or document['stars'] is None:
                continue
            for star in document['stars']:
                for term in star.split():
                    term_lower = term.lower()
                    if term_lower not in star_index:
                        star_index[term_lower] = {}
                    if document['id'] not in star_index[term_lower]:
                        star_index[term_lower][document['id']] = 1
                    else:
                        star_index[term_lower][document['id']] += 1
        return star_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """
        genre_index = {}
        for document in self.preprocessed_documents:
            if 'genres' not in document or document['genres'] is None:
                continue
            for genre in document['genres']:
                for term in genre.split():
                    term_lower = term.lower()
                    if term_lower not in genre_index:
                        genre_index[term_lower] = {}
                    if document['id'] not in genre_index[term_lower]:
                        genre_index[term_lower][document['id']] = 1
                    else:
                        genre_index[term_lower][document['id']] += 1
        return genre_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        current_index = {}
        for document in self.preprocessed_documents:
            if 'summaries' not in document or document['summaries'] is None:
                continue
            for summary in document['summaries']:
                for term in summary.split():
                    if term not in current_index:
                        current_index[term] = {}
                    if document['id'] not in current_index[term]:
                        current_index[term][document['id']] = 1
                    else:
                        current_index[term][document['id']] += 1
        return current_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            return list(self.index[index_type][word].keys())
        except:
            return []
        
    def add_to_index(self, index_value, text, document_id):
        index = self.index[index_value]
        for terms in text:
            for term in terms.split():
                if term not in index:
                    index[term] = {}
                if document_id not in index[term]:
                    index[term][document_id] = 1
                else:
                    index[term][document_id] += 1

    def remove_from_index(self, index_value, text, document_id):
        index = self.index[index_value]
        for terms in text:
            for term in terms.split():
                if term not in index:
                    continue
                if document_id not in index[term]:
                    continue
                del index[term][document_id]

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """
        document_id = document['id']
        if document_id in self.index[Indexes.DOCUMENTS.value]:
            return
        self.add_to_index(Indexes.STARS.value, document['stars'], document_id)
        self.add_to_index(Indexes.GENRES.value, document['genres'], document_id)
        self.add_to_index(Indexes.SUMMARIES.value, document['summaries'], document_id)
        self.index[Indexes.DOCUMENTS.value][document_id] = document

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """
        if document_id in self.index[Indexes.DOCUMENTS.value]:
            return
        document = self.index[Indexes.DOCUMENTS.value][document_id]
        self.remove_from_index(Indexes.STARS.value, document['stars'], document_id)
        self.remove_from_index(Indexes.GENRES.value, document['geners'], document_id)
        self.remove_from_index(Indexes.SUMMARIES.value, document['summaries'], document_id)
        del self.index[Indexes.DOCUMENTS.value][document_id]

    def delete_dummy_keys(self, index_before_add, index, key):
        if len(index_before_add[index][key]) == 0:
            del index_before_add[index][key]


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['tim', 'henry'],
            'genres': ['drama', 'crime'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if not index_before_add[Indexes.STARS.value].__contains__('tim'):
            index_before_add[Indexes.STARS.value].setdefault('tim', {})

        if (set(index_after_add[Indexes.STARS.value]['tim']).difference(set(index_before_add[Indexes.STARS.value]['tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, tim')
            return

        if not index_before_add[Indexes.STARS.value].__contains__('henry'):
            index_before_add[Indexes.STARS.value].setdefault('henry', {})

        if (set(index_after_add[Indexes.STARS.value]['henry']).difference(set(index_before_add[Indexes.STARS.value]['henry']))
                != {dummy_document['id']}):
            print('Add is incorrect, henry')
            return

        if not index_before_add[Indexes.GENRES.value].__contains__('drama'):
            index_before_add[Indexes.GENRES.value].setdefault('drama', {})

        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if not index_before_add[Indexes.GENRES.value].__contains__('crime'):
            index_before_add[Indexes.GENRES.value].setdefault('crime', {})

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if not index_before_add[Indexes.SUMMARIES.value].__contains__('good'):
            index_before_add[Indexes.SUMMARIES.value].setdefault('good', {})

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        # Change the index_before_remove to its initial form if needed

        self.delete_dummy_keys(index_before_add, Indexes.STARS.value, 'tim')
        self.delete_dummy_keys(index_before_add, Indexes.STARS.value, 'henry')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'drama')
        self.delete_dummy_keys(index_before_add, Indexes.GENRES.value, 'crime')
        self.delete_dummy_keys(index_before_add, Indexes.SUMMARIES.value, 'good')

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if not os.path.exists(path):
            os.makedirs(path)

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        save_path = os.path.join(path, f'{index_name}_index.json')
        with open(save_path, 'w') as f:
            json.dump(self.index[index_name], f)

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """
        index = {}
        for index_name in  [Indexes.DOCUMENTS.value, Indexes.STARS.value, Indexes.GENRES.value, Indexes.SUMMARIES.value]:
            save_path = os.path.join(path, f'{index_name}_index.json')
            with open(save_path, 'r') as f:
                data = json.load(f)
            index[index_name] = data
        return index

    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time <= brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
if __name__ == '__main__':
    with open('../utility/preprocessed.json', 'r') as f:
        preprocessed = json.load(f)
    index = Index(preprocessed)
    index.check_add_remove_is_correct()
    print('saving indexes...')
    for index_name in  [Indexes.DOCUMENTS.value, Indexes.STARS.value, Indexes.GENRES.value, Indexes.SUMMARIES.value]:
        index.store_index('./index', index_name)
    print('saved indexes')
    print('')
    print('loading indexes...')
    loaded_index = index.load_index('./index')
    for index_name in  [Indexes.DOCUMENTS.value, Indexes.STARS.value, Indexes.GENRES.value, Indexes.SUMMARIES.value]:
        print(f'checking {index_name}... result: {index.check_if_index_loaded_correctly(index_name, loaded_index[index_name])}')
    index.index = loaded_index
    print('loaded indexes')
    print('')
    print('checking indexing...')
    for index_name, check_word in  [(Indexes.DOCUMENTS.value, 'good'),
                                    (Indexes.DOCUMENTS.value, 'bad'),
                                    (Indexes.STARS.value, 'bachchan'),
                                    (Indexes.STARS.value, 'bernal'),
                                    (Indexes.GENRES.value, 'animation'),
                                    (Indexes.GENRES.value, 'family'),
                                    (Indexes.SUMMARIES.value, 'showrunner'),
                                    (Indexes.SUMMARIES.value, 'straitened')]:
        print(f'checking={index_name} with word={check_word}')
        index.check_if_indexing_is_good(index_name, check_word=check_word)
        print('')
