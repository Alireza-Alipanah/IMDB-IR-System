import os
os.environ["WANDB_DISABLED"] = "true"

from BertFinetuner_mask import BERTFinetuner

# file_path = stop_words_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utility', 'IMDB_crawled.json')
# Instantiate the class
bert_finetuner = BERTFinetuner('bert_data.json', top_n_genres=5)

# Load the dataset
bert_finetuner.load_dataset()

# Preprocess genre distribution
bert_finetuner.preprocess_genre_distribution()

# Split the dataset
bert_finetuner.split_dataset()

# Fine-tune BERT model
bert_finetuner.fine_tune_bert()

# Compute metrics
bert_finetuner.evaluate_model()

# Save the model (optional)
bert_finetuner.save_model('Movie_Genre_Classifier')