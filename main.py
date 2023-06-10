from package.utilities.statistics import Statistics
from package.RNN.preprocessing import Preprocessing


def main() -> None:
    preprocessing = Preprocessing("data\\News_Category_Dataset_v3.json")
    stat_obj = Statistics(preprocessing)
    stat_obj.display_dataset_statistics(save=True)


if __name__ == "__main__":
    main()
