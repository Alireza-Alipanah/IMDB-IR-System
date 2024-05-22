import json
from indexes_enum import Indexes,Index_types
from index_reader import Index_reader

class CollectionIndex:
    def __init__(self, path='index/'):
        """
        Initializes the DocumentLengthsIndex class.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.

        """
        self.documents_index = Index_reader(path, index_name=Indexes.DOCUMENTS).index
        self.document_length_index = {
            Indexes.STARS: self.get_collection_index(Indexes.STARS.value),
            Indexes.GENRES: self.get_collection_index(Indexes.GENRES.value),
            Indexes.SUMMARIES: self.get_collection_index(Indexes.SUMMARIES.value)
        }
        self.store_collection_index(path, Indexes.STARS)
        self.store_collection_index(path, Indexes.GENRES)
        self.store_collection_index(path, Indexes.SUMMARIES)

    def read_documents(self, crawled_data_path):
        """
        Reads the documents.
        
        """
        with open(crawled_data_path, 'r') as f:
            return json.load(f)

    def get_collection_index(self, where):
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
        total = 0
        cf = dict()
        lenghts = {}
        for document_id, docuemnt in self.documents_index.items():
            if docuemnt[where] is not None:
                for i in docuemnt[where]:
                    splt = i.split()
                    total += len(splt)
                    for j in splt:
                        if j in cf:
                            cf[j] += 1
                        else:
                            cf[j] = 1
        for i in cf.keys():
            cf[i] /= total
        return cf
    
    def store_collection_index(self, path , index_name):
        """
        Stores the document lengths index to a file.

        Parameters
        ----------
        path : str
            The path to the directory where the indexes are stored.
        index_name : Indexes
            The name of the index to store.
        """
        path = path + index_name.value + '_' + Index_types.COLLECTION.value + '_index.json'
        with open(path, 'w') as file:
            json.dump(self.document_length_index[index_name], file, indent=4)
    

if __name__ == '__main__':
    document_lengths_index = CollectionIndex()
    print('Collection index stored successfully.')