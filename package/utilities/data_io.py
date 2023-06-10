import pandas as pd
import os
from typing import Text


class DataIO:
    def __init__(self) -> None:
        pass

    @staticmethod
    def load_data(dataset_path: Text) -> None:
        try:
            return pd.read_json(dataset_path, lines=True)
        except Exception as ex:
            print(ex)

    @staticmethod
    def save_statistics(statistics: Text = "") -> None:
        if not os.path.exists("statistics"):
            os.mkdir("statistics")
        with open(os.path.join("statistics", "dataset_statistics.txt"), "w") as f:
            f.write(statistics)