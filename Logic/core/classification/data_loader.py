########## needed for relative import ##########
import inspect
import sys
import os

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
################################################

import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from word_embedding.fasttext_model import FastText, preprocess_text


class ReviewLoader:
    def __init__(self, file_path: str, fasttext_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []
        self.labelencoder = LabelEncoder()
        self.fasttext_path = fasttext_path


    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        data = pd.read_csv(self.file_path)
        data['review'] = data['review'].apply(preprocess_text)
        data['class'] = self.labelencoder.fit_transform(data['sentiment'])
        self.fasttext_model = FastText(None)
        self.fasttext_model.prepare(None, mode = "load", path=self.fasttext_path)
        self.data = data

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        self.data['embeddings'] = self.data['review'].apply(lambda x: self.fasttext_model.get_query_embedding(x, do_preprocess=False))
        return self.data['embeddings']

    def split_data(self, test_data_ratio=0.2):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(self.data['embeddings'].values.tolist(),
         self.data['class'].values.tolist(), test_size=test_data_ratio)
        return X_train, X_test, y_train, y_test
