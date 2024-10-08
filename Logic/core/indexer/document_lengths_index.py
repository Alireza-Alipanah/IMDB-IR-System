import json
try:
    from .indexes_enum import Indexes,Index_types
    from .index_reader import Index_reader
except ImportError:
    from indexes_enum import Indexes,Index_types
    from index_reader import Index_reader

class DocumentLengthsIndex:
    def __init__(self, path='index/', index_unique_words=False):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """
        self.index_unique_words = index_unique_words
        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        self.document_length_index = {
            Indexes.STARS: self.get_documents_length(Indexes.STARS.value),
            Indexes.GENRES: self.get_documents_length(Indexes.GENRES.value),
            Indexes.SUMMARIES: self.get_documents_length(Indexes.SUMMARIES.value)
        }
        self.store_document_lengths_index(path, Indexes.STARS)
        self.store_document_lengths_index(path, Indexes.GENRES)
        self.store_document_lengths_index(path, Indexes.SUMMARIES)

    def read_documents(self, crawled_data_path):
        """
        Reads the documents.
        
        """
        with open(crawled_data_path, 'r') as f:
            return json.load(f)

    def get_documents_length(self, where):
        """
        Gets the documents' length for the specified field.

        Parameters
        ----------
        where : str
            The field to get the document lengths for.

        Returns
        -------
        dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field (where).
        """
        lenghts = {}
        for document_id, docuemnt in self.documents_index.items():
            if docuemnt[where] is not None:
                if self.index_unique_words:
                    all_words = set()
                    for i in docuemnt[where]:
                        all_words.update(i.split())
                    lenghts[document_id] = len(all_words)
                else:
                    lenghts[document_id] = sum([len(i.split()) for i in docuemnt[where]])
        return lenghts
    
    def store_document_lengths_index(self, path , index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        if self.index_unique_words :
            path = path + index_name.value + '_' + Index_types.DOCUMENT_UNIQUE_LENGTH.value + '_index.json'
        else:
            path = path + index_name.value + '_' + Index_types.DOCUMENT_LENGTH.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)
    

if __name__ == '__main__':
    document_lengths_index = DocumentLengthsIndex()
    print('Document lengths index stored successfully.')
    document_lengths_index = DocumentLengthsIndex(index_unique_words=True)
    print('Document lengths unique words index stored successfully.')
