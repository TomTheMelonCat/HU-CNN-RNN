import os
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from PIL import Image
from typing import Dict, Any


class Statistics:
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def display_dataset_statistics(self, save: bool = True) -> None:
        stats = self.get_dataset_statistics()
        descriptions = {
            "num_instances": "Number of instances - the total number of images in the dataset.",
            "num_classes": "Number of classes - the total number of unique categories in the dataset.",
            "instances_per_class": "Instances per class - the number of images in each category.",
            "min_image_size": "Min image size - the dimensions (width, height) of the smallest image in the dataset.",
            "max_image_size": "Max image size - the dimensions (width, height) of the largest image in the dataset.",
            "mean_image_width": "Mean image width - the average width of the images in the dataset.",
            "std_image_width": "Std image width - the standard deviation of the image widths in the dataset.",
            "mean_image_height": "Mean image height - the average height of the images in the dataset.",
            "std_image_height": "Std image height - the standard deviation of the image heights in the dataset.",
            "mean_aspect_ratio": "Mean aspect ratio - the average aspect ratio (width/height) of the images in the dataset.",
            "most_common_aspect_ratio": "Most common aspect ratio - the most frequently occurring aspect ratio in the dataset.",
            "percent_portrait": "Percent portrait - the percentage of images that are portrait oriented (height > width).",
            "percent_landscape": "Percent landscape - the percentage of images that are landscape oriented (width > height).",
            "percent_square": "Percent square - the percentage of images that are exactly square (width == height)."
        }
        if not os.path.exists('statistics'):
            os.mkdir('statistics')
        with open(os.path.join('statistics', 'dataset_statistics.txt'), 'w') as f:
            for key, value in stats.items():
                if key in descriptions:
                    f.write(f"{descriptions[key]}\n{value}\n\n")
            f.write(f"The plot of class distributions is present in statistics folder under class_distributions.png name.\nFull path of the image is:\n")
            f.write(f"{os.path.join(Path(__file__).parents[2], 'statistics', 'class_distribution.png')}")
        self.plot_class_distribution(save)
        
    def get_dataset_statistics(self) -> Dict[str, Any]:
        stats = {}
        sizes = self.df['filename'].apply(lambda x: Image.open(x).size)
        aspect_ratios = sizes.apply(lambda x: x[0] / x[1])

        stats['num_instances'] = len(self.df)
        stats['num_classes'] = self.df['category'].nunique()
        stats['instances_per_class'] = self.df['category'].value_counts().to_dict()
        stats['min_image_size'] = min(sizes)
        stats['max_image_size'] = max(sizes)
        stats['mean_image_width'] = sizes.apply(lambda x: x[0]).mean()
        stats['std_image_width'] = sizes.apply(lambda x: x[0]).std()
        stats['mean_image_height'] = sizes.apply(lambda x: x[1]).mean()
        stats['std_image_height'] = sizes.apply(lambda x: x[1]).std()
        stats['mean_aspect_ratio'] = aspect_ratios.mean()
        stats['most_common_aspect_ratio'] = aspect_ratios.mode()[0]
        stats['percent_portrait'] = (aspect_ratios < 1).mean() * 100
        stats['percent_landscape'] = (aspect_ratios > 1).mean() * 100
        stats['percent_square'] = (aspect_ratios == 1).mean() * 100

        return stats

    def plot_class_distribution(self, save: bool) -> None:
        plt.figure(figsize=(10, 8))
        self.df['category'].value_counts().plot(kind='bar')
        plt.title('Class Distribution')
        plt.xlabel('Classes')
        plt.ylabel('Number of instances')
        if save:
            plt.savefig('statistics/class_distribution.png')
            plt.show()
        else:
            plt.show()
