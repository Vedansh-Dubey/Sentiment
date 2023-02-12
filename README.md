# Sentiment Analysis for Indic Languages (Marathi, Gujarati, Punjabi)

Sentiment analysis is the process of determining the emotional tone behind a series of words, used to gain an understanding of the attitudes, opinions, and emotions expressed within an online mention. This repository contains a collection of code, data, and resources for performing sentiment analysis on text data.

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

## License

This repository is licensed under the MIT License. See the [LICENSE]() file for details.
