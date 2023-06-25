import keras.layers as L
import numpy as np
import os

from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from typing import Any, Text

from CNN_basic_package.utilities.data_io import DataIO


class BasicCNN:
    def __init__(self, preprocessing: Any) -> None:
        self.params = DataIO.load_config("preprocessing", "training", "general")
        self.preprocessing = preprocessing

    def train_(self) -> None:
        model = Sequential(
            [
                L.Conv2D(
                    32,
                    (3, 3),
                    activation="relu",
                    input_shape=(self.params["preprocessing"].values()),
                ),
                L.BatchNormalization(),
                L.MaxPooling2D(pool_size=(2, 2)),
                L.Dropout(0.3),
                L.Conv2D(64, (3, 3), activation="relu"),
                L.BatchNormalization(),
                L.MaxPooling2D(pool_size=(2, 2)),
                L.Dropout(0.3),
                L.Conv2D(128, (3, 3), activation="relu"),
                L.BatchNormalization(),
                L.MaxPooling2D(pool_size=(2, 2)),
                L.Dropout(0.3),
                L.Conv2D(256, (3, 3), activation="relu"),
                L.BatchNormalization(),
                L.MaxPooling2D(pool_size=(2, 2)),
                L.Dropout(0.3),
                L.Conv2D(512, (3, 3), activation="relu"),
                L.BatchNormalization(),
                L.MaxPooling2D(pool_size=(2, 2)),
                L.Dropout(0.3),
                L.Flatten(),
                L.Dense(1024, activation="relu"),
                L.BatchNormalization(),
                L.Dropout(0.3),
                L.Dense(2, activation="softmax"),
            ]
        )

        optimizer = Adam(learning_rate=self.params["training"]["lr"])

        model.compile(
            loss="categorical_crossentropy", optimizer=optimizer, metrics=["acc"]
        )

        model.summary()

        early_stopping = EarlyStopping(
            monitor="val_loss",
            mode="min",
            verbose=1,
            patience=self.params["training"]["acc_patience"],
        )

        dynamic_lr = ReduceLROnPlateau(
            monitor="val_acc",
            patience=self.params["training"]["reduction_patience"],
            cooldown=self.params["training"]["reduction_cooldown"],
            verbose=1,
            factor=self.params["training"]["reduction_factor"],
            min_lr=self.params["training"]["minimum_lr"],
        )

        checkpointing = ModelCheckpoint(
            DataIO.get_model_path("basic_CNN"),
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        )

        history = model.fit(
            self.preprocessing.train_generator,
            epochs=self.params["training"]["epochs"],
            validation_data=self.preprocessing.validation_generator,
            validation_steps=self.preprocessing.validation_df.shape[0]
            // self.params["general"]["batch_size"],
            steps_per_epoch=self.preprocessing.train_df.shape[0]
            // self.params["general"]["batch_size"],
            callbacks=[early_stopping, dynamic_lr, checkpointing],
        )

    def test_(self) -> None:
        model = load_model(DataIO.get_model_path("basic_CNN"))

        test_steps = np.ceil(
            self.preprocessing.test_df.shape[0] / self.params["general"]["batch_size"]
        )

        test_loss, test_acc = model.evaluate(
            self.preprocessing.test_generator,
            steps=test_steps,
            verbose=0,
        )
        print(f"Model's loss and accuracy: {test_loss} {test_acc}")
