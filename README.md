# Sentiment Analysis for Indic Languages (Marathi, Gujarati, Punjabi)

Sentiment analysis is the process of determining the emotional tone behind a series of words, used to gain an understanding of the attitudes, opinions, and emotions expressed within an online mention. This repository contains a collection of code, data, and resources for performing sentiment analysis on text data.

## Data Preprocessing:

Data preprocessing is a crucial step in sentiment analysis as it helps in cleaning and transforming the raw data into a format that can be easily analyzed and interpreted. The following steps can be followed for data preprocessing in sentiment analysis:

- Data Collection: The first step is to collect the data from various sources such as online forums, social media platforms, or surveys.

- Data Cleaning: The next step is to clean the data by removing any irrelevant information such as special characters, numbers, and symbols. This step also involves removing duplicates, correcting typos, and handling missing values.

- Text Normalization: In this step, the text is converted into a standard format. This includes converting all the words to lowercase, stemming and lemmatization, and removing stop words.

- Data Transformation: In this step, the data is transformed into a numerical format that can be easily analyzed by machine learning algorithms. This can be done using techniques such as one-hot encoding, bag of words, or word embeddings.

- Data Splitting: The final step is to split the data into training and testing sets, with the training set being used to train the machine learning models, and the testing set being used to evaluate the performance of the models.

- It's important to note that data preprocessing is an iterative process, and multiple rounds of cleaning and normalization may be required to obtain the best results.

## Requirements
- Python 3.5+
- Tensorflow 2.x or PyTorch
- Numpy, Pandas

## Contents
- Pre-trained models: A set of pre-trained sentiment analysis models that can be used for sentiment analysis on new data.
- Data: Datasets used for training and evaluating sentiment analysis models.
- Notebooks: Jupyter notebooks with code for performing sentiment analysis, including training and evaluation of models and examples of how to use pre-trained models for sentiment analysis.
- App: A simple streamlit UI application to implement the pretrained model and obtain the sentimental analyzed output

## Installation

Use the package manager [pip] to install all required dependencies.

```bash
pip install -r requirements.txt
```

## Usage

- Clone the repository to your local machine.
- Install the required libraries using the command `pip install -r requirements.txt`
- Load a pre-trained model of your choice.
- Use the model to perform sentiment analysis on new data by passing it a string of text.
- Run `streamlit run app.py` to run the sample interactive UI application

## Training your own Models

1) Clone the repository to your local machine.
2) Install the required libraries using the command `pip install -r requirements.txt`
3) Split the data into training and test sets.
4) Train a sentiment analysis model on the training data.
5) Evaluate the model on the test data.
6) Use the model to perform sentiment analysis on new data by passing it a string of text.

## Contributions
This repository is open to contributions. If you have any suggestions for improving the sentiment analysis models or have a new model you'd like to add, feel free to create a pull request.


## Citing this project:

If you use this project in your research or wish to refer to the results, please use the following citation:

```bash
@misc{Sentiment Analysis,
  author = {Vedansh-Dubey},
  title = {Indic language sentiment analysis},
  year = {2023},
  howpublished = {\url{https://github.com/Vedansh-Dubey/Sentiment}}
}
```

## License

This repository is licensed under the MIT License. See the [LICENSE]() file for details.
