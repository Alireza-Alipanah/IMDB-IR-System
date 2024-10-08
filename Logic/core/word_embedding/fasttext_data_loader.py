import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        df = pd.read_json(self.file_path)
        df = df[['synposis', 'summaries', 'reviews', 'title', 'genres']]
        return df
        

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        df['synposis'] = df['synposis'].apply(lambda x: ' '.join(x) if x is not None else '')
        df['summaries'] = df['summaries'].apply(lambda x: ' '.join(x) if x is not None else '')
        df['reviews'] = df['reviews'].apply(lambda x: ' '.join([' '.join(i) for i in x]) if x is not None else '')
        df['X'] = df[['synposis', 'summaries', 'reviews', 'title']].agg(' '.join, axis=1)
        label_encoder = LabelEncoder()
        label_encoder.fit(df['genres'].explode().unique())
        df['genre_encoded'] = df['genres'].apply(label_encoder.transform)
        X = df['X'].values.tolist()
        y = df['genre_encoded'].values.tolist()
        texts = df['synposis'].values.tolist() + df['summaries'].values.tolist() + df['reviews'].values.tolist() + df['title'].values.tolist()
        return X, y, texts


