import numpy as np
from sklearn.metrics import classification_report
from sklearn.svm import SVC

try:
    from .basic_classifier import BasicClassifier
    from .data_loader import ReviewLoader
except ImportError:
    from basic_classifier import BasicClassifier
    from data_loader import ReviewLoader

import os


class SVMClassifier(BasicClassifier):
    def __init__(self):
        super().__init__()
        self.model = SVC()

    def fit(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc
        """
        self.model.fit(x, y)

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
        return self.model.predict(x)

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


# F1 accuracy : 78%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    fasttext_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'word_embedding', 'FastText_model.bin')
    data_loader = ReviewLoader(file_path='IMDB_Dataset.csv', fasttext_path=fasttext_model_path)
    data_loader.load_data()
    data_loader.get_embeddings()
    X_train, X_test, y_train, y_test = data_loader.split_data()

    classifier = SVMClassifier()
    classifier.fit(X_train, y_train)
    report = classifier.prediction_report(X_test, y_test)
    print(report)
