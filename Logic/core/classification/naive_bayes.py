import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    from .basic_classifier import BasicClassifier
    from .data_loader import ReviewLoader
except ImportError:
    from basic_classifier import BasicClassifier
    from data_loader import ReviewLoader

import os


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        x = self.cv.fit_transform(x)
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.number_of_samples = x.shape[0]
        self.number_of_features = x.shape[1]

        self.prior = np.bincount(y) / self.number_of_samples

        self.feature_probabilities = {}
        for cls in self.classes:
            mask = (y == cls)
            x_cls = x[mask]
            feature_counts = np.sum(x_cls, axis=0)
            total_count = np.sum(feature_counts)
            self.feature_probabilities[cls] = np.asarray(np.log((feature_counts + self.alpha) / (total_count + self.alpha * self.number_of_features)))
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        x = self.cv.transform(x)
        log_probs = np.zeros((x.shape[0], self.num_classes))
        for i, cls in enumerate(self.classes):
            for j in range(x.shape[0]):
                log_probs[j, i] = np.sum(self.feature_probabilities[cls] * x[j, :].toarray())
            log_probs[:, i] += np.log(self.prior[i])

        return np.argmax(log_probs, axis=1)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        return classification_report(y, y_pred)

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        prediction = self.predict(sentences)
        return np.sum(prediction) / prediction.shape[0]


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviwes using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    path ='IMDB_Dataset.csv'
    data = pd.read_csv(path)
    data['class'] = data['sentiment'].apply(lambda x: int(x == 'positive'))

    cv = CountVectorizer(max_features=100000, max_df=0.8)
    X_train, X_test, y_train, y_test = train_test_split(data['review'].values.tolist(), data['class'].values.tolist(), test_size=0.1)
    nb = NaiveBayes(cv)
    nb.fit(X_train, y_train)
    print(nb.prediction_report(X_test, y_test))

