import pandas as pd
import os

from pathlib import Path
from typing import Text, Any


class DataIO:
    @staticmethod
    def load_data(dataset_path: Text) -> None:
        try:
            return pd.read_json(dataset_path, lines=True)
        except Exception as ex:
            print(ex)

    @staticmethod
    def save_statistics(statistics: Text = "", root_path: Text = "") -> None:
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        with open(os.path.join(root_path, "dataset_statistics.txt"), "w") as f:
            f.write(statistics)

    @staticmethod
    def save_model_weights(model: Any, folder_name: Text = "model") -> None:
        model_path = os.path.join(Path(__file__).parents[2], folder_name)

        if not os.path.exists(model_path):
            os.mkdir(model_path)

        model.save_weights(os.path.join(model_path, "RNN_custom_weights.h5"))
        model.load_weights(os.path.join(model_path, "RNN_custom_weights.h5"))
        model.save(os.path.join(model_path, "RNN_custom_model.hdf5"))