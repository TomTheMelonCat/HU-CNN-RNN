import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict, List, Any, Text, Type
from package.utilities.data_io import DataIO


class Statistics:
    def __init__(self, preprocessing: Any) -> None:
        self.preprocessing = preprocessing

    def display_dataset_statistics(self, save: bool = True) -> None:
        self.save = save
        statistics = '\n'.join([str(arg) for arg in self.get_dataset_statistics()])
        print(statistics)
        if save:
            DataIO.save_statistics(statistics=statistics)
        self.plot_class_distribution()

    def get_dataset_statistics(self) -> None:
        full_path = os.path.join(Path(__file__).parents[2], "statistics", "class_distributions.png")
        statistics = [
            f"Dataset length: {len(self.preprocessing.data)}\n",
            f"Dataset schema: \n{self.preprocessing.data.dtypes}\n",
            f"Data preview (10 rows): \n{self.preprocessing.data.head(1).transpose()}\n",
            f"Missing values: \n{self.preprocessing.data.isna().sum()}\n",
            f"Train length: {len(self.preprocessing.train_df)}",
            f"Test length: {len(self.preprocessing.test_df)}\n",
            f"Class distributions of the dataset are represented as a plot saved at:\n{full_path}."
        ]

        return statistics

    def plot_class_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(13, 6))
        fig.canvas.manager.set_window_title('Distribution')

        value_counts = self.preprocessing.data['category'].value_counts(normalize=True).sort_index()
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
        if self.save:
            plt.savefig(os.path.join("statistics", "class_distributions"))

        plt.show()
