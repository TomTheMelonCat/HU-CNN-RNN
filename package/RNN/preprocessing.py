import json
import os
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Any, Text
from sklearn.model_selection import train_test_split



class Preprocessing:
    def __init__(self, dataset_path: Text = None, statistics: bool = True, save_statistics: bool = True, save_path: Text = "statistics") -> None:
        if dataset_path is None:
            raise("Data path must be specified - only .JSON format is supported.")
        self.dataset_path = dataset_path
        self.statistics = statistics
        self.save_statistics = save_statistics if statistics else False
        self.save_path = save_path
        self.load_data()
        self.set_train_test()
        self.set_dataset_statistics()
        self.display_dataset_statistics()
        self.save_dataset_statistics()


    def load_data(self) -> None:
        data = pd.read_json(self.dataset_path, lines=True)
        self.data = data
    

    def set_train_test(self) -> None:
        self.train_df, self.test_df = train_test_split(self.data, test_size=0.2, random_state=42, stratify=self.data['category'])


    def set_dataset_statistics(self) -> None:
        if not self.statistics:
            return
        
        full_path = f"{Path(__file__).parents[2]}\{self.save_path}\class_distributions.png"
        self.statistics = [
            f"Dataset length: {len(self.data)}\n",
            f"Data preview (10 rows): {self.data.head(10)}\n",
            f"Missing values: {self.data.isna().sum()}\n",
            f"Train length: {len(self.train_df)}",
            f"Test length: {len(self.test_df)}\n",
            f"Class distributions of the dataset are represented as a plot saved at:\n{full_path}."
        ]


    def display_dataset_statistics(self) -> None:
        to_print = '\n'.join([str(arg) for arg in self.statistics])
        print(to_print)
        if self.save_statistics:
            self.save_dataset_statistics(to_print)
        self.plot_class_distribution()
    

    def save_dataset_statistics(self, content: Text = "") -> None:
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        with open(os.path.join(self.save_path, "dataset_statistics.txt"), "w") as f:
            f.write(content)


    def plot_class_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(18,8))
        fig.canvas.manager.set_window_title('Distribution')

        value_counts = self.train_df['category'].value_counts(normalize=True).sort_index()
        value_counts.plot(kind='bar', ax=ax, width=0.5)

        for p in ax.patches:
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy() 
            ax.text(x+width/2, 
                    y+height, 
                    '{:.2%}'.format(height), 
                    horizontalalignment='center',
                    verticalalignment='bottom', 
                    fontsize=8,)  
            
        ax.set_xlabel('Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion', fontsize=12, fontweight='bold')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, horizontalalignment='right')

        ax.set_title('Dataset class distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)  # Adjust the top padding of the figure
        if self.save_statistics:
            plt.savefig(os.path.join(self.save_path, "class_distributions"))

        plt.show()

prp = Preprocessing("data\\News_Category_Dataset_v3.json")