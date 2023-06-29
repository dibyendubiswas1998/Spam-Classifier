import re
import string

import numpy as np
import pandas as pd

# NLP:
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# TF-IDF & Bag of Words:
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_absolute_error, confusion_matrix
from src.utils.common_utils import log

# download: stopwords, punctuation, words, etc.
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download("wordnet")
# nltk.download('words')


def text_preprocessing(text, log_file):
    """
        text preprocessing:\n
        :param text: text data
        :param log_file: log_file
        :return: preprocessed data
    """
    try:
        log(file_object=log_file, log_message=f"text pre-processing start")  # logs the details

        # lowering the text
        text = text.lower()

        # remove the numbers
        text = re.sub(r'\d+', '', text)

        # removing the punctuation
        translator = str.maketrans('', '', string.punctuation)
        text = text.translate(translator)

        # tokenize the text to word
        tokens = word_tokenize(text)

        # Remove the stop-of-words:
        stop_of_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_of_words]

        # Stemming the data:
        porter = PorterStemmer()
        porter_stem = [porter.stem(word) for word in tokens]

        return " ".join(porter_stem)  # return text after text pre-processing.


    except Exception as ex:
        log(file_object=log_file, log_message=f"Error will be: {ex}")  # logs the details
        print(ex)
        raise ex


def word2vector(text, log_file):
    """
        Convert the text to vector using TF-IDF.\n
        :param text: text
        :param log_file: log_file
        :return: vector
    """
    try:
        log(file_object=log_file, log_message=f"convert the text to vector using TF-IDF")  # logs the details
        tfidf = TfidfVectorizer(max_features=1500)
        vector = tfidf.fit_transform(text).toarray()
        return vector

    except Exception as ex:
        log(file_object=log_file, log_message=f"Error will be: {ex}")  # logs the details
        print(ex)
        raise ex


def load_and_split_data(data_path, log_file):
    """
        load the pre-processed data\n
        :param data_path: data_path
        :param log_file: log_file
        :return: x_train y_train x_text y_text
    """
    try:
        log(file_object=log_file, log_message=f"load & split the data")  # logs the details
        data = pd.read_csv(data_path)
        X = data.drop("target", axis=1)  # drop target feature
        Y = data['target']

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=0)  # split data
        return x_train, x_test, y_train, y_test

    except Exception as ex:
        log(file_object=log_file, log_message=f"Error will be: {ex}")  # logs the details
        print(ex)
        raise ex


def model_creation(x_train, y_train, log_file):
    """
        Create Models.\n
        :param log_file: log_file
        :param x_train: x_train
        :param y_train: y_train
    """
    try:
        log(file_object=log_file, log_message=f"model creation is started.")  # logs the details
        model = MultinomialNB()
        model.fit(x_train, y_train)    # fit the model:
        return model

    except Exception as ex:
        log(file_object=log_file, log_message=f"Error will be: {ex}")  # logs the details
        print(ex)
        raise ex
    

def model_performance(x_train, x_test, y_train, y_test, model, log_file):
    """
        Check the model performance.\n
        :param x_train: x_train
        :param x_test: x_text
        :param y_train: y_train
        :param y_test: y_test
        :param model: model
        :param log_file: log_file
        :return: performance_report
    """
    try:
        log(file_object=log_file, log_message=f"Evaluate the model performance")  # logs the details
        # based on train data:
        predict_train = model.predict(x_train)
        recall_train = recall_score(y_true=y_train, y_pred=predict_train)
        precision_train = precision_score(y_true=y_train, y_pred=predict_train)
        accuracy_train = accuracy_score(y_true=y_train, y_pred=predict_train)
        mae_train = mean_absolute_error(y_true=y_train, y_pred=predict_train)
        confusion_matrix_train = confusion_matrix(y_true=y_train, y_pred=predict_train)

        # based on test data
        predict_test = model.predict(x_test)
        recall_test = recall_score(y_true=y_test, y_pred=predict_test)
        precision_test = precision_score(y_true=y_test, y_pred=predict_test)
        accuracy_test = accuracy_score(y_true=y_test, y_pred=predict_test)
        mae_test = mean_absolute_error(y_true=y_test, y_pred=predict_test)
        confusion_matrix_test = confusion_matrix(y_true=y_test, y_pred=predict_test)

        # report:
        report = {
            "For Training data " : {
                 "recall_score": recall_train,
                 "precision_score": precision_train,
                 "accuracy_score": accuracy_train,
                 "mean_absolute_error_score": mae_train,
                 # "confusion_matrix": list(confusion_matrix_train),
             },
            "For Test data" : {
                "recall_score": recall_test,
                "precision_score": precision_test,
                "accuracy_score": accuracy_test,
                "mean_absolute_error_score": mae_test,
                # "confusion_matrix": list(confusion_matrix_test),
            }
        }
        return report

    except Exception as ex:
        log(file_object=log_file, log_message=f"Error will be: {ex}")  # logs the details
        print(ex)
        raise ex


