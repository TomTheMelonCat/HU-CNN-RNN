from pandas import DataFrame
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.resnet import preprocess_input
from sklearn.model_selection import train_test_split
from typing import Text

from CNN_package.utilities.statistics import Statistics
from CNN_package.utilities.data_io import DataIO


class Preprocessing:
    def __init__(self, train_val_df: DataFrame, test_df: DataFrame, display_statistics: bool = True, save_statistics: bool = True, mode: Text = "default") -> None:
        self.train_val_df = train_val_df
        self.test_df = test_df
        self.mode = mode
        if self.mode == "default":
            self.params = DataIO.load_config("preprocessing", "general")
        elif self.mode.lower() == "resnet":
            self.params = DataIO.load_config("preprocessing_resnet", "general")
            self.params["preprocessing"] = self.params.pop("preprocessing_resnet")

        if display_statistics:
            self.statistics = Statistics(train_val_df)
            if save_statistics:
                DataIO.save_statistics("placeholder_for_statistics")
        self.split_data()
        self.set_generators()

    def split_data(self) -> None:
        self.train_df, self.validation_df = train_test_split(self.train_val_df, test_size=0.15, random_state=42, stratify=self.train_val_df['category'])
        self.train_df = self.train_df.reset_index(drop=True)
        self.validation_df = self.validation_df.reset_index(drop=True)

    def set_generators(self) -> None:
        if self.mode.lower() == "resnet":
            train_img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
            validation_img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
            testing_img_gen = ImageDataGenerator(preprocessing_function=preprocess_input)
        else:
            train_img_gen = ImageDataGenerator(
                rotation_range=15,
                rescale=1./255,
                shear_range=0.1,
                zoom_range=0.2,
                horizontal_flip=True,
                width_shift_range=0.15,
                height_shift_range=0.15
            )
            validation_img_gen = ImageDataGenerator(rescale=1./255)
            testing_img_gen = ImageDataGenerator(rescale=1./255)

        self.train_generator = train_img_gen.flow_from_dataframe(
            self.train_df, 
            "../input/train/train/", 
            x_col='filename',
            y_col='category',
            target_size=(self.params["preprocessing"]["img_heigth"], self.params["preprocessing"]["img_width"]),
            class_mode='categorical',
            batch_size=self.params["general"]["batch_size"]
        )

        self.validation_generator = validation_img_gen.flow_from_dataframe(
            self.validation_df, 
            "../input/train/train/", 
            x_col='filename',
            y_col='category',
            target_size=(self.params["preprocessing"]["img_heigth"], self.params["preprocessing"]["img_width"]),
            class_mode='categorical',
            batch_size=self.params["general"]["batch_size"]
        )

        self.test_generator = testing_img_gen.flow_from_dataframe(
            self.test_df, 
            "../input/test1/test1/", 
            x_col='filename',
            y_col=None,
            class_mode=None,
            target_size=(self.params["preprocessing"]["img_heigth"], self.params["preprocessing"]["img_width"]),
            batch_size=self.params["general"]["batch_size"],
            shuffle=False
        )