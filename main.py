import os

from CNN_package.utilities.data_io import DataIO
from CNN_package.CNN.preprocessing import Preprocessing
from CNN_package.CNN.custom_CNN import CustomCNN
from CNN_package.CNN.custom_ResNet import CustomResnet

ROOT = os.path.dirname(__file__)

def main() -> None:
    data_train_val = DataIO.load_data(os.path.join(ROOT, "data", "train"))
    data_test = DataIO.load_data(os.path.join(ROOT, "data", "test"), True)

    #preprocessing = Preprocessing(data_train_val, data_test, True, True, "default")
    #custom_cnn = CustomCNN(preprocessing)
    #custom_cnn.train_()
    #custom_cnn.test_()

    preprocessing = Preprocessing(data_train_val, data_test, True, True, "resnet")
    custom_resnet = CustomResnet(preprocessing)
    custom_resnet.train_()
    custom_resnet.test_()


if __name__ == "__main__":
    main()