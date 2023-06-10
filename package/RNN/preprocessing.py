import pandas as pd
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.preprocessing.text import Tokenizer
from typing import Dict, List, Any, Text
from sklearn.model_selection import train_test_split

from package.utilities.data_io import DataIO


class Preprocessing:
    def __init__(self, dataset_path: Text = None) -> None:
        if dataset_path is None:
            print("Data path must be specified - only .JSON format is supported.")
            return -1
        self.data = DataIO.load_data(dataset_path)
        self.set_train_test()
        self.transform_data()

    def set_train_test(self) -> None:
        self.train_df, self.test_df = train_test_split(self.data, test_size=0.2, random_state=42, stratify=self.data['category'])

    def transform_data(self) -> None:
        self.train_df["headline_processed"] = self.train_df["headline"].apply(self.preprocess)
        self.test_df["headline_processed"] = self.test_df["headline"].apply(self.preprocess)

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(self.train_df["headline_processed"])
        self.train_df["sequences"] = tokenizer.texts_to_sequences(self.train_df["headline_processed"])
        vocab = len(tokenizer.word_index)+1
        print(vocab)

    def preprocess(self, text) -> Text:
        stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)

        words = " ".join([stemmer.stem(word) for word in text.split() if word not in stop_words])

        return words
