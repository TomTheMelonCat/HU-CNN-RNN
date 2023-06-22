import keras.layers as L
import numpy as np

from keras.applications import ResNet50
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from typing import Any, Text

from CNN_package.utilities.data_io import DataIO


class CustomResnet:
    def __init__(self, preprocessing: Any) -> None:
        self.params = DataIO.load_config("preprocessing_resnet", "training", "general")
        self.params["preprocessing"] = self.params.pop("preprocessing_resnet")
        self.preprocessing = preprocessing

    def train_(self) -> None:
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=tuple(self.params["preprocessing"].values()))

        # Freeze the layers of the base model
        for layer in base_model.layers:
            layer.trainable = False

        model = Sequential(
            [
                base_model,
                L.Flatten(),
                L.Dense(1024, activation="relu"),
                L.BatchNormalization(),
                L.Dropout(0.2),
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
            DataIO.get_model_path(), save_best_only=True, monitor="val_loss", mode="min"
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
        model = load_model(DataIO.get_model_path())

        test_steps = np.ceil(self.preprocessing.test_df.shape[0] / self.params["general"]["batch_size"])

        predictions = model.predict(self.preprocessing.test_generator, steps=test_steps)
        
        print("Sample Predictions:")
        for i in range(20):
            image_path = self.preprocessing.test_generator.filepaths[i]
            image_filename = os.path.basename(image_path)
            print(f"Prediction for {image_filename}: {predictions[i]}")