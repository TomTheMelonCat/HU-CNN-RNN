import pandas as pd
import re
import nltk
import numpy as np

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from num2words import num2words
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from typing import Dict, List, Any, Text
from sklearn.model_selection import train_test_split

from RNN_package.utilities.data_io import DataIO
from RNN_package.utilities.statistics import Statistics



nltk.download('stopwords')

class Preprocessing:
    def __init__(self, dataset_path: Text = None, statistics: bool = True, save_statistics: bool = True, n_classes: int = 15) -> None:
        if dataset_path is None:
            print("Data path must be specified - only .JSON format is supported.")
            return -1
        self.data = DataIO.load_data(dataset_path)
        self.n_classes = n_classes
        if statistics:
            stat_obj = Statistics(self)
            stat_obj.display_dataset_statistics(save=save_statistics)
        self.transform_data()
        self.set_train_test()

    def transform_data(self) -> None:

        self.data.loc[self.data['category']=='PARENTING', 'category'] = 'PARENTS'
        self.data.loc[self.data['category']=='THE WORLDPOST', 'category'] = 'WORLD NEWS'
        self.data.loc[self.data['category']=='BLACK VOICES', 'category'] = 'VOICES'
        self.data.loc[self.data['category']=='QUEER VOICES', 'category'] = 'VOICES'
        self.data.loc[self.data['category']=='WEDDINGS', 'category'] = 'WEDDINGS & DIVORCE'
        self.data.loc[self.data['category']=='DIVORCE', 'category'] = 'WEDDINGS & DIVORCE'
        self.data.loc[self.data['category']=='HEALTHY LIVING', 'category'] = 'WELLNESS'
        self.data.loc[self.data['category']=='COMEDY', 'category'] = 'ENTERTAINMENT'

        self.data = self.data.sample(frac=1).reset_index(drop=True) # row shuffling
        self.data = self.data[~self.data['short_description'].apply(lambda x: len(x)==0)]
        valid_counts = self.data['category'].value_counts()[0:self.n_classes].index.tolist()
        self.data = self.data[self.data['category'].isin(valid_counts)].reset_index(drop=True)
        self.data['input'] = self.data.apply(lambda x: str(x['headline']) + ' ' + str(x['short_description']), axis=1)
        self.data['input_processed']= self.data["input"].apply(self.preprocess)

        self.categories_mask = {k:v for v,k in enumerate(self.data["category"].unique())}

    def preprocess(self, text) -> Text:
        #stemmer = PorterStemmer()
        stop_words = set(stopwords.words('english'))

        text = text.lower()
        text= re.sub('[^a-zA-Z0-9\.]', ' ', str(text).lower())
        text = re.sub('\s+', ' ', text)
        text = text.replace('.','')
        text = re.sub('\n', ' ', text)

        nums = re.findall('[0-9]+\.?[0-9]*', text)
        try:
            for num in nums:
                text = text.replace(num, num2words(num))
        except Exception as e:
            print('exception', e)
            print(f'for {num} in {text}')

        words = " ".join([word for word in text.split() if word not in stop_words])

        return words

    def set_train_test(self) -> None:
        self.tokenizer = Tokenizer()
        self.train_df, self.test_df = train_test_split(self.data, test_size=0.05, random_state=42, stratify=self.data['category'])
        self.train_df["category"].replace(self.categories_mask, inplace=True)
        self.test_df["category"].replace(self.categories_mask, inplace=True)
        self.categories_train = self.train_df["category"]
        self.categories_test = self.test_df["category"]
        self.__tokenize__(self.train_df)
        self.vocab_size = len(self.tokenizer.word_index) + 1 

        self.sequences_train = self.tokenizer.texts_to_sequences(self.train_df["input_processed"])
        self.sequences_test = self.tokenizer.texts_to_sequences(self.test_df["input_processed"]) 

        sequence_lengths = [len(seq) for seq in self.sequences_train]
        max_length = int(np.percentile(sequence_lengths, 90))
        self.sequences_train = pad_sequences(self.sequences_train, maxlen=max_length, padding='post', truncating='post')
        self.sequences_test = pad_sequences(self.sequences_test, maxlen=max_length, padding='post', truncating='post')

        self.sequences_train = pad_sequences(self.sequences_train, padding='post')
        self.sequences_test = pad_sequences(self.sequences_test, padding='post')

    def __tokenize__(self, data: pd.DataFrame) -> None:
        self.tokenizer.fit_on_texts(data["input_processed"])