import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

try:
    from .basic_classifier import BasicClassifier
    from .data_loader import ReviewLoader
except ImportError:
    from basic_classifier import BasicClassifier
    from data_loader import ReviewLoader

import os
import multiprocessing as mp


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors, use_mp=True):
        super().__init__()
        self.k = n_neighbors
        self.use_mp = use_mp

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

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
        self.X_train = x
        self.y_train = y

    def predict_one_doc(self, xi):
        scores = [(float('inf'), -1)]
        for j in range(len(self.X_train)):
            scores.append((np.linalg.norm(xi - self.X_train[j]), self.y_train[j]))
            scores.sort(key=lambda x: x[0])
            scores = scores[:self.k]
        scores = np.array([i[1] for i in scores])
        return np.bincount(scores).argmax()

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
        if self.use_mp:
            with mp.Pool(16) as pool:
                predictions = pool.map(self.predict_one_doc, x)
        else:
            predictions = []
            for xi in x:
                predictions.append(predict_one_doc(xi))
        return np.array(predictions)

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


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    fasttext_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'word_embedding', 'FastText_model.bin')
    data_loader = ReviewLoader(file_path='IMDB_Dataset.csv', fasttext_path=fasttext_model_path)
    data_loader.load_data()
    data_loader.get_embeddings()
    X_train, X_test, y_train, y_test = data_loader.split_data()

    classifier = KnnClassifier(n_neighbors=3)
    classifier.fit(X_train, y_train)
    report = classifier.prediction_report(X_test, y_test)
    print(report)
