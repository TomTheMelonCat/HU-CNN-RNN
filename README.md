# HU-CNN-RNN
Creation and testing of several CNN and RNN models in Python via Tensorflow

## CNN Project Information 🠔
- You can get 10 to 20 points
- Every project must include brief description of the dataset
    - Number of images, number of classes and class balance
    - Examples of the images
- What metrics scores you have decided to use
    - e.g. Accuracy, Precision, Recall, F1-score etc.
    - Also state which one of the scores is the most important from your point of view given the class balance, task, ...
- Try minimally 2 different models
    - The first model should be built from scratch, i.e. create your own architecture and train the model
        - If you perform various experiments or parameters tuning be sure to include everyting in the Notebook step by step with some brief comments about the experiments (e.g. effect of BatchNorm/layer sizes/optimizer on accuracy/train time/...)
    - The second model will employ transfer learning techniques
        - You can try any model you wish, e.g. ResNet, Inception, MobileNet, etc.
        - Take a look at this list and try at least one of them
        - Fine tune the model for your dataset and compare it with the first one
- Mandatory part of every project is a summary at the end in which you summarize the most interesting insight obtained.

- Estimated time for the project is 3-6h, this value heavily depends on your skill, but you can use it as a guidance for a project size.

- Result is a Jupyter Notebook with descriptions included or a PDF report + source codes.

## RNN Project Information 🠔
- You can get 10 to 20 points
- Every project must include brief description of the dataset
    - Number of instances, number of classes and class balance
    - Examples of the text data
- What metrics scores you have decided to use
    - e.g. Accuracy, Precision, Recall, F1-score etc.
    - Also state which one of the scores is the most important from your point of view given the class balance, task, ...
- Try minimally 2 different models

    - The first model should be built from scratch, i.e. create your own architecture and train the model
        - If you perform various experiments or parameters tuning be sure to include everyting in the Notebook step by step with some brief comments about the experiments (e.g. effect of BatchNorm/layer sizes/optimizer on accuracy/train time/...)
    - The second model will employ transfer learning techniques
        - Use any set of pre-trained embedding vectors (GloVe, Word2Vec, FastText etc.) or any transformer-based model (this is optional as it is more advanced approach above this course complexity)
        - Fine tune the model for your dataset and compare it with the first one
- Mandatory part of every project is a summary at the end in which you summarize the most interesting insight obtained.

- Result is a Jupyter Notebook with descriptions included or a PDF report + source codes.

## Final Project Information 🠔
- You can get 20 to 50 points

- Goal of the project is to create more thorough analysis of the chosen dataset than in the previous two smaller projects.

- The dataset selection is up to you however is has come from image, text or time series domain.

- Every project must include brief description of the dataset
    - Number of instances, number of classes and class balance
    - Examples of the data,
    - ...
- What metrics scores you have decided to use
    - e.g. Accuracy, Precision, Recall, F1-score etc.
    - Also state which one of the scores is the most important from your point of view given the class balance, task, ...
- Try at least 3 different models
    - The first two models (one simple and one more complex) should be built from scratch, i.e. create your own architecture and train the models
    - The third model will employ transfer learning techniques
        - Use any set of pre-trained embedding vectors (GloVe, Word2Vec, FastText etc.) or any transformer-based model (this is optional as it is more advanced approach above this course complexity) or pre-trained network in case of image dataset
    - The project will include hyper parameter tuning so try different batch sizes, optimizers etc. and document everything accordingly
- Mandatory part of every project is a summary at the end in which you summarize the most interesting insight obtained.