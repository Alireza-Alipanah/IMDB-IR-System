import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re


def get_stopwords():
    with open('stopwords.txt', 'r') as f:
        return [i.strip() for i in f.readlines()]

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        self.documents = documents
        self.stopwords = set(get_stopwords())
        self.lemmatizer = WordNetLemmatizer()

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
        preprocessed_documents = []
        for document in self.documents:
            document = self.remove_links(document)
            document = self.remove_punctuations(document)
            document = self.remove_stopwords(document)
            document = self.normalize(' '.join(document))
            preprocessed_documents.append(document)
        return preprocessed_documents

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        text = text.lower()
        words = self.tokenize(text)
        words = [self.lemmatizer.lemmatize(word) for word in words]
        return ' '.join(words)

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        for pattern in patterns:
            text = re.sub(pattern, ' ', text)
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        return re.sub(r'[^\w\s]', ' ', text)

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if not word in self.stopwords]
        return filtered_text
    

if __name__ == '__main__':
    import json

    with open('IMDB_crawled.json', 'r') as f:
        crawled = json.load(f)
    preprocessed = []
    for document in crawled:
        if document['summaries'] is None:
            continue
        preprocessor = Preprocessor(document['summaries'])
        document['summaries'] = preprocessor.preprocess()
        preprocessed.append(document)
    with open('preprocessed.json', 'w') as f:
            json.dump(preprocessed, f)

