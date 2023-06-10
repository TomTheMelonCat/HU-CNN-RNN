import json
import os
import pandas as pd
import nltk
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Any, Text
from sklearn.model_selection import train_test_split
from package.utilities.data_io import DataIO



class Statistics:
    def __init__(self, data: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        self.data = data
        self.train_df = train_df
        self.test_df = test_df
        self.set_dataset_statistics()

    def display_dataset_statistics(self, save: bool = True) -> None:
        to_print = '\n'.join([str(arg) for arg in self.statistics])
        print(to_print)
        if save:
            DataIO.save_statistics()
        self.plot_class_distribution()

    def set_dataset_statistics(self) -> None:
        full_path = f"{Path(__file__).parents[2]}\{self.save_path}\class_distributions.png"
        statistics = [
            f"Dataset length: {len(self.data)}\n",
            f"Data preview (10 rows): {self.data.head(10)}\n",
            f"Missing values: {self.data.isna().sum()}\n",
            f"Train length: {len(self.train_df)}",
            f"Test length: {len(self.test_df)}\n",
            f"Class distributions of the dataset are represented as a plot saved at:\n{full_path}."
        ]

        return statistics

    def plot_class_distribution(self, data: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(18,8))
        fig.canvas.manager.set_window_title('Distribution')

        value_counts = data['category'].value_counts(normalize=True).sort_index()
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