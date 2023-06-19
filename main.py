from RNN_package.RNN.preprocessing import Preprocessing
from RNN_package.RNN.custom_RNN import CustomRNN
from RNN_package.RNN.custom_BERT import CustomBERT


def main() -> None:
    preprocessing = Preprocessing("data\\News_Category_Dataset_v3.json", True, True)
    # Custom created RNN
    custom_rnn = CustomRNN(
        preprocessing.vocab_size,
        preprocessing.categories_train,
        preprocessing.categories_test,
        preprocessing.sequences_train,
        preprocessing.sequences_test,
        preprocessing.n_classes
    )
    custom_rnn.train_()

    # Custom RNN utilizing BERT
    custom_bert = CustomBERT(
        preprocessing.categories_train,
        preprocessing.categories_test,
        preprocessing.train_df["input_processed"],
        preprocessing.test_df["input_processed"],
        preprocessing.n_classes
    )
    custom_bert.train_()

if __name__ == "__main__":
    main()
