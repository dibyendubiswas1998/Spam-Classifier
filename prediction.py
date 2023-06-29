import pandas as pd
import joblib
from src.spam_classifier import text_preprocessing
from src.utils.common_utils import log, clean_prev_dirs_if_exis, create_dir


def prediction(new_data):
    """
        Prediction based on new data.\n
        :param new_data: new_data
        :return: result
    """
    try:
        log_file = "artifacts/Logs/logs.txt"
        log(file_object=log_file, log_message="prediction process is start.")   # logs the details

        # text preprocessing:
        preprocessed_text = text_preprocessing(text=new_data, log_file=log_file)

        # transform the vector using TF-IDF:
        vectorizer_path = "artifacts/Vectorizer/vectorizer.joblib"
        tfidf = joblib.load(vectorizer_path)
        vector = tfidf.transform([preprocessed_text])


        # # load the model:
        model_path = "artifacts/Model/model.joblib"
        model = joblib.load(model_path)

        # # predict result:
        result = model.predict(vector)
        return "Not Spam" if result[0] == 0 else "Spam"


    except Exception as ex:
        log_file = "artifacts/Logs/logs.txt"
        log(file_object=log_file, log_message=f"Error will be: {ex}")  # logs the details
        print(ex)
        raise ex
