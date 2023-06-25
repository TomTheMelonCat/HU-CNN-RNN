import os

from CNN_basic_package.utilities.data_io import DataIO
from CNN_basic_package.preprocessing import Preprocessing
from CNN_basic_package.basic_CNN import BasicCNN
from CNN_basic_package.transfer_CNN import TransferCNN
from CNN_basic_package.utilities.statistics import Statistics

ROOT = os.path.dirname(__file__)

def main() -> None:
    data_train_val = DataIO.load_data(os.path.join(ROOT, "data", "horse-or-human", "train"))
    data_test = DataIO.load_data(os.path.join(ROOT, "data", "horse-or-human", "validation"))
    statistics = Statistics(data_train_val)
    statistics.display_dataset_statistics()

    preprocessing = Preprocessing(data_train_val, data_test, True, True, "default")
    simple_cnn = BasicCNN(preprocessing)
    simple_cnn.train_()
    simple_cnn.test_()

    preprocessing = Preprocessing(data_train_val, data_test, True, True, "inception")
    custom_inception = TransferCNN(preprocessing)
    custom_inception.train_()
    custom_inception.test_()


if __name__ == "__main__":
    main()