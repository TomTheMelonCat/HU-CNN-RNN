import os
import pandas as pd
import matplotlib.pyplot as plt

from pandas import DataFrame
from pathlib import Path
from typing import Dict, List, Any, Text, Type


class Statistics:
    def __init__(self, df: DataFrame) -> None:
        self.df = df

    def display_dataset_statistics(self, save: bool = True) -> None:
        return True

    def get_dataset_statistics(self) -> None:
        return True

    def plot_class_distribution(self) -> None:
        fig, ax = plt.subplots(figsize=(18, 8))
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
            plt.savefig(os.path.join("statistics", "class_distributions.png"))

        plt.show()
