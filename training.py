from src.spam_classifier import load_and_split_data, model_creation, model_performance
from src.utils.common_utils import log, create_dir, clean_prev_dirs_if_exis, save_model, save_report



def training():
    """
        These helps to train the model
        :return: None
    """
    try:
        log_file = "artifacts/Logs/logs.txt"
        log(file_object=log_file, log_message="training process is start.")   # logs the details

        # load & split the data:
        preprocessed_data_dir = "artifacts/Preprocessed_Data/data.csv"
        x_train, x_test, y_train, y_test = load_and_split_data(data_path=preprocessed_data_dir,
                                                               log_file=log_file)

        # model creation & save the model:
        model_dir = "artifacts/Model"
        model_path = "artifacts/Model/model.joblib"
        model = model_creation(x_train=x_train, y_train=y_train, log_file=log_file)
        clean_prev_dirs_if_exis(dir_path=model_dir)  # remove directory if already present
        create_dir(dirs=[model_dir])  # create artifacts/Model directory
        save_model(model_name=model, model_path=model_path)  # save the model

        # model performance:
        report_dir = "artifacts/Performance_Report"
        report_file_name = "artifacts/Performance_Report/report.json"
        report = model_performance(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test,
                                   model=model, log_file=log_file)
        clean_prev_dirs_if_exis(dir_path=report_dir)  # remove directory if already present
        create_dir(dirs=[report_dir])  # create artifacts/Performance_Report directory
        save_report(file_path=report_file_name, report=report)  # save the report
        log(file_object=log_file, log_message="successfully training process is completed\n\n")  # logs the details



    except Exception as ex:
        log_file = "artifacts/Logs/logs.txt"
        log(file_object=log_file, log_message=f"Error will be: {ex}")  # logs the details
        print(ex)
        raise ex

