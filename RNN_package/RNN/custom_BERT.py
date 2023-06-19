import os
import numpy as np
import tensorflow_text
import tensorflow_hub as hub
import tensorflow as tf
import keras.layers as L

from pathlib import Path
from keras.losses import SparseCategoricalCrossentropy
from keras.optimizers import Adam
from typing import List, Text
from sklearn.utils import class_weight
from RNN_package.utilities.data_io import DataIO


class CustomBERT:
    def __init__(
        self,
        train_labels: List[int],
        test_labels: List[int],
        train_text: List[Text],
        test_text: List[Text],
        n_classes: int,
    ) -> None:
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

        preprocessor_path = os.path.join(Path(__file__).parents[2], "BERT_small", "preprocessing")
        encoder_path = os.path.join(Path(__file__).parents[2], "BERT_small", "encoding")

        bert_preprocess = hub.KerasLayer(preprocessor_path)
        bert_encoder = hub.KerasLayer(encoder_path, trainable = True)

        # BERT layers
        text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='bert')
        preprocessed_text = bert_preprocess(text_input)
        outputs = bert_encoder(preprocessed_text)

        # hyper parameters
        EPOCHS = 25
        BATCH_SIZE = 64
        units = 128


        sequential_layers = tf.keras.Sequential(
            [
                L.Bidirectional(L.LSTM(units, activation="tanh",
                                recurrent_activation="sigmoid",
                                recurrent_dropout=0,
                                unroll=False,
                                use_bias=True,
                                return_sequences=True)),
                L.BatchNormalization(),
                L.GlobalAveragePooling1D(),
                L.Dense(256, activation="relu"),
                L.Dropout(0.4),
                L.Dense(128, activation="relu"),
                L.Dropout(0.4),
                L.Dense(self.n_classes),
            ]
        )

        model_outputs = sequential_layers(outputs['sequence_output'])
        model = tf.keras.Model(inputs=text_input, outputs=model_outputs)

        optimizer = Adam(learning_rate=0.0001)

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
            monitor="val_loss", mode="min", verbose=1, patience=4
        )
        model.summary()
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

        DataIO.save_model_weights(model=model, folder_name="model_weights_BERT")
