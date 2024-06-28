import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, EvalPrediction
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn


class BERTFinetuner:
    """
    A class for fine-tuning the BERT model on a movie genre classification task.
    """

    def __init__(self, file_path, top_n_genres=5):
        """
        Initialize the BERTFinetuner class.

        Args:
            file_path (str): The path to the JSON file containing the dataset.
            top_n_genres (int): The number of top genres to consider.
        """
        self.file_path = file_path
        self.top_n_genres = top_n_genres
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=top_n_genres, problem_type="multi_label_classification")
        self.dataset = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.top_n_genres_list = None
        self.mlb = None

    def load_dataset(self):
        """
        Load the dataset from the JSON file.
        """
        self.dataset = pd.read_json(self.file_path)

    def preprocess_genre_distribution(self):
        """
        Preprocess the dataset by filtering for the top n genres
        """
        genre_counter = Counter([genre for genres in self.dataset['genres'] for genre in genres])
        top_genres = genre_counter.most_common(self.top_n_genres)
        top_genres = [genre for genre, count in top_genres]

        self.dataset = self.dataset[self.dataset['genres'].apply(lambda x: any(genre in top_genres for genre in x))]
        self.dataset['genres'] = self.dataset['genres'].apply(lambda x: [i for i in x if i in top_genres])

        genre_distribution = Counter([genre for genres in self.dataset['genres'] for genre in genres])
        genres, counts = zip(*genre_distribution.items())
        self.top_n_genres_list = genres

        plt.figure(figsize=(10, 6))
        sns.barplot(x=list(genres), y=list(counts))
        plt.title('Genre Distribution')
        plt.ylabel('Count')
        plt.xlabel('Genre')
        plt.xticks(rotation=45)
        plt.show()

    def split_dataset(self, test_size=0.2, val_size=0.1):
        """
        Split the dataset into train, validation, and test sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            val_size (float): The proportion of the dataset to include in the validation split.
        """
        train_val_data, test_data = train_test_split(self.dataset, test_size=test_size, random_state=42)
        train_data, val_data = train_test_split(train_val_data, test_size=val_size/(1-test_size), random_state=42)

        self.train_dataset = train_data
        self.val_dataset = val_data
        self.test_dataset = test_data

    def create_dataset(self, encodings, labels):
        """
        Create a PyTorch dataset from the given encodings and labels.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.

        Returns:
            IMDbDataset: A PyTorch dataset object.
        """
        return IMDbDataset(encodings, labels)

    def fine_tune_bert(self, epochs=5, batch_size=16, warmup_steps=500, weight_decay=0.01):
        """
        Fine-tune the BERT model on the training data.

        Args:
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training.
            warmup_steps (int): The number of warmup steps for the learning rate scheduler.
            weight_decay (float): The strength of weight decay regularization.
        """
        train_texts = self.train_dataset['first_page_summary'].tolist()
        val_texts = self.val_dataset['first_page_summary'].tolist()

        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True)
        val_encodings = self.tokenizer(val_texts, truncation=True, padding=True)

        self.mlb = MultiLabelBinarizer(classes=self.top_n_genres_list)  # Ensure classes match top_n_genres
        train_labels = self.mlb.fit_transform(self.train_dataset['genres'])
        val_labels = self.mlb.transform(self.val_dataset['genres'])

        train_dataset = self.create_dataset(train_encodings, train_labels)
        val_dataset = self.create_dataset(val_encodings, val_labels)

        training_args = TrainingArguments(
            output_dir='./results',
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            logging_dir='./logs',
            logging_steps=10,
            evaluation_strategy='epoch',
            save_strategy='epoch',
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics
        )

        trainer.train()

    def compute_metrics(self, pred):
        """
        Compute evaluation metrics based on the predictions.

        Args:
            pred (EvalPrediction): The model's predictions.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        labels = pred.label_ids
        preds = (pred.predictions > 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
        acc = accuracy_score(labels, preds)

        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }

    def evaluate_model(self):
        """
        Evaluate the fine-tuned model on the test set.
        """
        test_texts = self.test_dataset['first_page_summary'].tolist()
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True)
        
        test_labels = self.mlb.transform(self.test_dataset['genres'])

        test_dataset = self.create_dataset(test_encodings, test_labels)
        trainer = Trainer(model=self.model, compute_metrics=self.compute_metrics)
        metrics = trainer.evaluate(eval_dataset=test_dataset)
        print(metrics)

    def save_model(self, model_name):
        """
        Save the fine-tuned model and tokenizer to the Hugging Face Hub.

        Args:
            model_name (str): The name of the model on the Hugging Face Hub.
        """
        self.model.save_pretrained(model_name)
        self.tokenizer.save_pretrained(model_name)

class IMDbDataset(torch.utils.data.Dataset):
    """
    A PyTorch dataset for the movie genre classification task.
    """

    def __init__(self, encodings, labels):
        """
        Initialize the IMDbDataset class.

        Args:
            encodings (dict): The tokenized input encodings.
            labels (list): The corresponding labels.
        """
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
            idx (int): The index of the item to retrieve.

        Returns:
            dict: A dictionary containing the input encodings and labels.
        """
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).float()
        return item

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: The number of items in the dataset.
        """
        return len(self.labels)
