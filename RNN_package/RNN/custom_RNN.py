import numpy as np
import tensorflow as tf
import keras.layers as L

from RNN_package.utilities.data_io import DataIO

from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam

from sklearn.utils import class_weight
from typing import List, Text



class CustomRNN:
    def __init__(
        self,
        vocabulary_count: int,
        train_labels: List[int],
        test_labels: List[int],
        train_text: List[Text],
        test_text: List[Text],
        n_classes: int,
    ) -> None:
        self.vocabulary_count = vocabulary_count
        self.train_labels = train_labels
        self.test_labels = test_labels
        self.train_text = train_text
        self.test_text = test_text
        self.n_classes = n_classes

    def train_(self) -> None:
        tf.keras.backend.clear_session()

        self.train_text = np.array(self.train_text)
        self.train_labels = np.array(self.train_labels)
        self.test_text = np.array(self.test_text)
        self.test_labels = np.array(self.test_labels)

        indices = np.arange(self.train_text.shape[0])
        np.random.shuffle(indices)

        self.train_text = self.train_text[indices]
        self.train_labels = self.train_labels[indices]

        # hyper parameters
        EPOCHS = 25 #20
        BATCH_SIZE = 512
        embedding_dim = 200
        units = 128

        model = tf.keras.Sequential(
            [
                L.Embedding(
                    self.vocabulary_count,
                    embedding_dim,
                    input_length=len(self.train_text[0]),
                ),
                L.Bidirectional(L.LSTM(units, return_sequences=True)),
                L.Bidirectional(L.LSTM(int(units/2), return_sequences=True)),
                L.GlobalMaxPooling1D(),
                L.Dense(256, activation="relu"),
                L.Dropout(0.3),
                L.Dense(128, activation="relu"),
                L.Dropout(0.3),
                L.Dense(64, activation="relu"),
                L.Dropout(0.2),
                L.Dense(self.n_classes),
            ]
        )

        optimizer = Adam(learning_rate=0.001)

        model.compile(
            loss=SparseCategoricalCrossentropy(from_logits=True),
            optimizer=optimizer,
            metrics=["acc"],
        )

        class_weights = class_weight.compute_class_weight(
            class_weight="balanced",
            classes=np.unique(self.train_labels),
            y=self.train_labels,
        )
        class_weights = dict(enumerate(class_weights))

        # Early Stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=6
        )

        history = model.fit(
            self.train_text,
            self.train_labels,
            epochs=EPOCHS,
            validation_split=0.15,
            batch_size=BATCH_SIZE,
            shuffle=False,
            callbacks=[early_stopping],
            class_weight=class_weights,
        )

        test_loss, test_acc = model.evaluate(
            self.test_text, self.test_labels, verbose=0
        )
        print(f"Test loss and accuracy: {test_loss}, {test_acc}")

        DataIO.save_model_weights(model=model, folder_name="model_weights")
