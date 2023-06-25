import json
import pandas as pd
import os

from pathlib import Path
from typing import Text, Any, Dict, Tuple



class DataIO:
    @staticmethod
    def load_data(dataset_path: Text, test: bool = False) -> pd.DataFrame:
        filenames = [str(child.resolve()) for child in Path.iterdir(Path(dataset_path))]
        if test:
            return pd.DataFrame({
            'filename': filenames,
        })
        
        categories = ["cat" if str(x).find("cat") != -1 else "dog" for x in filenames]
        return pd.DataFrame({
            'filename': filenames,
            'category': categories
        })
    
    @staticmethod
    def load_config(*keys: Tuple[Text]) -> Dict:
        with open(os.path.join(Path(__file__).parents[2], "config.json"), "r", encoding="utf-8") as read:
            data = json.load(read)
            try:
                return {k: data[k] for k in keys}
            except ValueError:
                print(f"Keys not resolved, returning entire configuration dictionary.")
                return data

    @staticmethod
    def get_model_path(folder_name: Text) -> Text:
        model_path_folder = os.path.join(Path(__file__).parents[2], folder_name)
        if not os.path.exists(model_path_folder):
            os.mkdir(model_path_folder)
        return os.path.join(model_path_folder, '.custom_CNN_best.hdf5')