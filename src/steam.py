import json

import joblib
import pandas as pd

from config import config
from utils import Logger


class Steam:
    def __init__(self, data, log_name, estimator=None):
        self.logger = self._get_logger(log_name)
        self.data: pd.DataFrame = self._get_data(data)
        self.estimator = self._get_estimator(estimator)
        self.temp = self._get_temp_json()

    @staticmethod
    def _get_logger(log_name):
        logger = Logger(log_name).get_logger()
        logger.info(f"logger {log_name} online")
        return logger

    def _get_data(self, data):
        try:
            df = pd.read_csv(config[data], sep="\t")
            self.logger.info("data load " + data)
        except Exception as e:
            self.logger.error("data loading error")
            self.logger.error("=" * 60)
            self.logger.error(e)
            self.logger.error("=" * 60)
            exit()
        else:
            return df

    def _get_estimator(self, estimator_name):
        if estimator_name is None:
            self.logger.info("estimator is None")
            return None
        try:
            file = estimator_name + ".pkl"
            estimator = joblib.load(config["model_path"] / file)
            self.logger.info("estimator load " + file)
        except Exception as e:
            self.logger.error("estimator load error")
            self.logger.error("=" * 60)
            self.logger.error(e)
            self.logger.error("=" * 60)
            exit()
        else:
            return estimator

    def _get_temp_json(self):
        try:
            with open(config["json_data"], "r") as f:
                j = json.load(f)
                self.logger.info("temp json loaded")
        except FileNotFoundError:
            self.logger.info("temp json init")
            j = {"models": [], "model_path": {}, "MSE": {}}
            with open(config["json_data"], "w") as f:
                json.dump(j, f)
        except Exception as e:
            self.logger.error("temp json load error")
            self.logger.error("=" * 60)
            self.logger.error(e)
            self.logger.error("=" * 60)
            exit()
        return j

    def write_temp_json(self):
        with open(config["json_data"], "w") as f:
            json.dump(self.temp, f)
        self.logger.info("temp json saved")

    def add_model(self, model, mse=None):
        model_str = str(model)
        str_end = model_str.find("(")
        model_str = model_str[:str_end]
        model_path = str(config["model_path"] / model_str) + ".pkl"
        joblib.dump(model, model_path)
        self.logger.info("adding model " + model_str)
        if model not in self.temp["models"]:
            self.temp["models"].append(model_str)
            self.temp["MSE"][model_str] = mse
            self.temp["model_path"][model_str] = model_path
            self.logger.info("=" * 60)
            self.logger.info("model add " + model_str)
            self.logger.info("MSE : " + str(mse))
            self.logger.info("model_path : " + model_path)
            self.logger.info("=" * 60)
            self.logger.info("model" + model_str + "added")
            self.write_temp_json()
        else:
            self.logger.info("model" + model_str + " already exists")

    def set_model(self, model, mse=None):
        model_str = str(model)
        str_end = model_str.find("(")
        model_str = model_str[:str_end]
        model_path = str(config["model_path"] / model_str) + ".pkl"
        if model_str in self.temp["models"]:
            self.logger.info("reset model " + model_str)
            self.temp["MSE"][model_str] = mse
            self.temp["model_path"][model_str] = model_path
            self.logger.info("=" * 60)
            self.logger.info("MSE : " + str(mse))
            self.logger.info("model_path : " + model_path)
            self.logger.info("=" * 60)
        else:
            self.add_model(model, mse=mse)

    def set_best_model(self):
        estimator_name = min(self.temp["MSE"], key=self.temp["MSE"].get)
        self.logger.info("set best model " + estimator_name)
        self.estimator = self._get_estimator(estimator_name)
