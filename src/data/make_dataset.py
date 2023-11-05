import pandas as pd
import string
import re
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import BertTokenizer


class DataProcessor:
    def __init__(self, file_path, tokenizer, dataset_size, batch_size=32):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.dataset_size = dataset_size
        self.batch_size = batch_size

    def _load_data(self):
        # Load the dataset into a pandas DataFrame
        data = pd.read_csv(self.file_path, sep='\t')
        return data[:self.dataset_size]

    def _preprocess_data(self, data):
        # Download necessary NLTK resources
        nltk.download('stopwords')
        nltk.download('wordnet')

        # Convert text to lowercase
        data['reference'] = data['reference'].str.lower()
        data['translation'] = data['translation'].str.lower()

        # Remove punctuation
        punc_table = str.maketrans('', '', string.punctuation)
        data['reference'] = data['reference'].apply(lambda x: x.translate(punc_table))
        data['translation'] = data['translation'].apply(lambda x: x.translate(punc_table))

        # Remove numbers
        data['reference'] = data['reference'].apply(lambda x: re.sub(r'\d+', '', x))
        data['translation'] = data['translation'].apply(lambda x: re.sub(r'\d+', '', x))

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        data['reference'] = (
            data['reference'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        )
        data['translation'] = (
            data['translation'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))
        )

        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        data['reference'] = (
            data['reference'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        )
        data['translation'] = (
            data['translation'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
        )

        return data

    def _tokenize_data(self, data):
        # Tokenize the data
        tokenized_data = self.tokenizer(
            data['reference'].tolist(),
            data['translation'].tolist(),
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=512,
            truncation='longest_first',
            padding='max_length',
            return_tensors='pt'
        )
        return tokenized_data

    def _create_labels(self, data):
        # Create labels based on the `ref_tox` column
        labels = (data['ref_tox'] > 0.5).astype(int).values
        return torch.tensor(labels)

    def _split_data(self, tokenized_data, labels):
        # Unpack the tokenized_data dictionary
        input_ids = tokenized_data['input_ids']
        attention_masks = tokenized_data['attention_mask']

        # Split the dataset into training and validation sets
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            input_ids, attention_masks, labels, test_size=0.1, random_state=1
        )

        return train_inputs, train_masks, val_inputs, val_masks, train_labels, val_labels

    def _create_data_loaders(self, train_inputs, train_masks, val_inputs, val_masks,
                             train_labels, val_labels, batch_size=32):
        # Create torch datasets
        train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
        val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=24)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=24)

        return train_loader, val_loader

    def process(self):
        # This method runs the entire data processing pipeline
        data = self._load_data()
        data = self._preprocess_data(data)
        tokenized_data = self._tokenize_data(data)
        labels = self._create_labels(data)
        train_inputs, train_masks, val_inputs, val_masks, train_labels, val_labels = (
            self._split_data(tokenized_data, labels)
        )
        train_loader, val_loader = self._create_data_loaders(
            train_inputs, train_masks, val_inputs, val_masks, train_labels, val_labels
        )
        return train_loader, val_loader
