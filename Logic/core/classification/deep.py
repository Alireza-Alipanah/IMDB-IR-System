import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from .basic_classifier import BasicClassifier
    from .data_loader import ReviewLoader
except ImportError:
    from basic_classifier import BasicClassifier
    from data_loader import ReviewLoader

import os


class ReviewDataSet(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)

        if len(self.embeddings) != len(self.labels):
            raise Exception("Embddings and Labels must have the same length")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.embeddings[i], self.labels[i]


class MLPModel(nn.Module):
    def __init__(self, in_features=100, num_classes=2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_features, 2048),
            nn.ReLU(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, xb):
        return self.network(xb)


class DeepModelClassifier(BasicClassifier):
    def __init__(self, in_features, num_classes, batch_size, x, y, num_epochs=50):
        """
        Initialize the model with the given in_features and num_classes
        Parameters
        ----------
        in_features: int
            The number of input features
        num_classes: int
            The number of classes
        batch_size: int
            The batch size of dataloader
        """
        super().__init__()
        self.test_loader = None
        self.in_features = in_features
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.model = MLPModel(in_features=in_features, num_classes=num_classes)
        self.best_model = self.model.state_dict()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.device = 'mps' if torch.backends.mps.is_available else 'cpu'
        self.device = 'cuda' if torch.cuda.is_available() else self.device
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        self.x = x
        self.y = y
        self.split_data()

    def split_data(self, test_data_ratio=0.2):
        X_train, X_test, y_train, y_test = train_test_split(self.x, self.y, test_size=test_data_ratio)
        X_validation, X_test, y_validation, y_test = train_test_split(X_test, y_test, test_size=0.5)
        train_dataset = ReviewDataSet(X_train, y_train)
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_dataset = ReviewDataSet(X_validation, y_validation)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataset = ReviewDataSet(X_test, y_test)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

    def fit(self):
        """
        Fit the model on the given train_loader and test_loader for num_epochs epochs.
        You have to call set_test_dataloader before calling the fit function.
        Parameters
        ----------
        x: np.ndarray
            The training embeddings
        y: np.ndarray
            The training labels
        Returns
        -------
        self
        """
        best_model_loss = float('inf')
        best_model = None
        self.prediction_report()
        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            running_loss = 0.0
            for xb, yb in self.train_dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
            avg_loss, _, _, _ = self._eval_epoch(self.validation_dataloader)
            if avg_loss < best_model_loss:
                best_model_loss = avg_loss
                best_model = self.model.state_dict()

        self.best_model = self.model.load_state_dict(best_model)

    def predict(self, x):
        """
        Predict the labels on the given test_loader
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        Returns
        -------
        predicted_labels: list
            The predicted labels
        """
        self.model.eval()
        x = torch.from_numpy(x).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def _eval_epoch(self, dataloader: torch.utils.data.DataLoader):
        """
        Evaluate the model on the given dataloader. used for validation and test
        Parameters
        ----------
        dataloader: torch.utils.data.DataLoader
        Returns
        -------
        eval_loss: float
            The loss on the given dataloader
        predicted_labels: list
            The predicted labels
        true_labels: list
            The true labels
        f1_score_macro: float
            The f1 score on the given dataloader
        """
        self.model.eval()
        all_preds, all_true_labels = [], []
        running_loss = 0.0
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                outputs = self.model(xb)
                loss = self.criterion(outputs, yb)
                running_loss += loss.item()
            
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_true_labels.extend(yb.cpu().numpy())

        avg_loss = running_loss / len(dataloader)
        f1_score_macro = f1_score(all_true_labels, all_preds, average='macro')

        return avg_loss, all_preds, all_true_labels, f1_score_macro

    def set_test_dataloader(self):
        """
        Set the test dataloader. This is used to evaluate the model on the test set while training
        Parameters
        ----------
        X_test: np.ndarray
            The test embeddings
        y_test: np.ndarray
            The test labels
        Returns
        -------
        self
            Returns self
        """
        return self._eval_epoch(self.test_dataloader)

    def prediction_report(self):
        """
        Get the classification report on the given test set
        Parameters
        ----------
        x: np.ndarray
            The test embeddings
        y: np.ndarray
            The test labels
        Returns
        -------
        str
            The classification report
        """
        avg_loss, all_preds, all_true_labels, f1_score_macro = self.set_test_dataloader()
    
        print("Classification Report:")
        print(classification_report(all_true_labels, all_preds))
        print(f"F1 Score Macro Average: {f1_score_macro:.4f}")
        return classification_report(all_true_labels, all_preds)

# F1 Accuracy : 79%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    fasttext_model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'word_embedding', 'FastText_model.bin')
    data_loader = ReviewLoader(file_path='IMDB_Dataset.csv', fasttext_path=fasttext_model_path)
    data_loader.load_data()
    data_loader.get_embeddings()

    classifier = DeepModelClassifier(
        in_features=data_loader.data['embeddings'][0].shape[0],
        num_classes=data_loader.data['class'].nunique(),
        batch_size=2048,
        x=np.array(data_loader.data['embeddings'].values.tolist()),
        y=np.array(data_loader.data['class'].values.tolist())
        )
    classifier.fit()
    report = classifier.prediction_report()
    print(report)
