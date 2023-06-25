import os

from CNN_package.utilities.data_io import DataIO
from CNN_package.CNN.preprocessing import Preprocessing
from CNN_package.CNN.simple_CNN import SimpleCNN
from CNN_package.CNN.complex_CNN import ComplexCNN
from CNN_package.CNN.custom_ResNet import CustomResnet
from CNN_package.utilities.statistics import Statistics

ROOT = os.path.dirname(__file__)

def main() -> None:
    data_train_val = DataIO.load_data(os.path.join(ROOT, "data", "train"))
    data_test = DataIO.load_data(os.path.join(ROOT, "data", "test"), True)
    statistics = Statistics(data_train_val)
    statistics.display_dataset_statistics()

    preprocessing = Preprocessing(data_train_val, data_test, True, True, "default")
    simple_cnn = SimpleCNN(preprocessing)
    simple_cnn.train_()
    simple_cnn.test_()

    complex_cnn = ComplexCNN(preprocessing)
    complex_cnn.train_()
    complex_cnn.test_()

    preprocessing = Preprocessing(data_train_val, data_test, True, True, "resnet")
    custom_resnet = CustomResnet(preprocessing)
    custom_resnet.train_()
    custom_resnet.test_()


if __name__ == "__main__":
    main()