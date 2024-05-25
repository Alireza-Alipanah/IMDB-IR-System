import fasttext
import re
import os
import multiprocessing as mp

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.spatial import distance
import nltk
nltk.download('stopwords')

try:
    from .fasttext_data_loader import FastTextDataLoader
except ImportError:
    from fasttext_data_loader import FastTextDataLoader


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()
    if punctuation_removal:
        text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [token for token in tokens if len(token) >= minimum_length]
    if stopword_removal:
        english_stopwords = set(stopwords.words('english'))
        all_stopwords = english_stopwords.union(set(stopwords_domain))
        tokens = [token for token in tokens if token not in all_stopwords]
    return ' '.join(tokens)


class FastText:
    """
    A class used to train a FastText model and generate embeddings for text data.

    Attributes
    ----------
    method : str
        The training method for the FastText model.
    model : fasttext.FastText._FastText
        The trained FastText model.
    """

    def __init__(self, preprocessor, method='skipgram', model_path=None, use_mp=True):
        """
        Initializes the FastText with a preprocessor and a training method.

        Parameters
        ----------
        method : str, optional
            The training method for the FastText model.
        """
        self.preprocessor = preprocessor
        self.method = method
        self.model = None
        self.use_mp = use_mp


    def train(self, texts):
        """
        Trains the FastText model with the given texts.

        Parameters
        ----------
        texts : list of str
            The texts to train the FastText model.
        """
        if self.use_mp:
            with mp.Pool(16) as pool:
                preprocessed_text = pool.map(self.preprocessor, texts)
        else:
            preprocessed_text = [self.preprocessor(i) for i in texts]
        with open('train_data.txt', 'w') as f:
            f.writelines(preprocessed_text)
        self.model = fasttext.train_unsupervised('train_data.txt', model=self.method)
        os.remove('train_data.txt')


    def get_query_embedding(self, query, do_preprocess=True):
        """
        Generates an embedding for the given query.

        Parameters
        ----------
        query : str
            The query to generate an embedding for.
        tf_idf_vectorizer : sklearn.feature_extraction.text.TfidfVectorizer
            The TfidfVectorizer to transform the query.
        do_preprocess : bool, optional
            Whether to preprocess the query.

        Returns
        -------
        np.ndarray
            The embedding for the query.
        """
        if do_preprocess:
            query = ' '.join(self.preprocessor(query))
        return self.model.get_sentence_vector(query)

    def analogy(self, word1, word2, word3):
        """
        Perform an analogy task: word1 is to word2 as word3 is to __.

        Args:
            word1 (str): The first word in the analogy.
            word2 (str): The second word in the analogy.
            word3 (str): The third word in the analogy.

        Returns:
            str: The word that completes the analogy.
        """
        vector_word1 = self.model[word1]
        vector_word2 = self.model[word2]
        vector_word3 = self.model[word3]

        relationship_vector = vector_word2 - vector_word1
        target_vector = vector_word3 + relationship_vector

        closest_word = None
        best_similarity = 0
        input_words = set([word1, word2, word3])
        for word in self.model.get_words():
            if word in input_words:
                continue
            word_vector = self.model[word]
            similarity = 1 - distance.cosine(target_vector, word_vector)

            if similarity > best_similarity:
                best_similarity = similarity
                closest_word = word

        return closest_word

    def save_model(self, path='FastText_model.bin'):
        """
        Saves the FastText model to a file.

        Parameters
        ----------
        path : str, optional
            The path to save the FastText model.
        """
        self.model.save_model(path)

    def load_model(self, path="FastText_model.bin"):
        """
        Loads the FastText model from a file.

        Parameters
        ----------
        path : str, optional
            The path to load the FastText model.
        """
        self.model = fasttext.load_model(path)

    def prepare(self, dataset, mode, save=False, path='FastText_model.bin'):
        """
        Prepares the FastText model.

        Parameters
        ----------
        dataset : list of str
            The dataset to train the FastText model.
        mode : str
            The mode to prepare the FastText model.
        """
        if mode == 'train':
            self.train(dataset)
        if mode == 'load':
            self.load_model(path)
        if save:
            self.save_model(path)

if __name__ == "__main__":
    ft_model = FastText(preprocessor=preprocess_text, method='skipgram')

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utility', 'IMDB_crawled.json')
    ft_data_loader = FastTextDataLoader(path)

    X, y, texts = ft_data_loader.create_train_data()

    # ft_model.train(texts)
    # ft_model.prepare(None, mode = "save", save=True)

    ft_model.prepare(None, mode = "load")

    print(10 * "*" + "Similarity" + 10 * "*")
    word = 'queen'
    neighbors = ft_model.model.get_nearest_neighbors(word, k=5)

    for neighbor in neighbors:
        print(f"Word: {neighbor[1]}, Similarity: {neighbor[0]}")

    print(10 * "*" + "Analogy" + 10 * "*")
    word1 = "king"
    word2 = "man"
    word3 = "queen"
    print(f"Similarity between {word1} and {word2} is like similarity between {word3} and {ft_model.analogy(word1, word2, word3)}")
