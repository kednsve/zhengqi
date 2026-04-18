import joblib
import pandas as pd

from config import config
from utils import Logger


class Steam:
    def __init__(self, data, estimator, log_name):
        self.logger = Logger(log_name).get_logger()
        self.data = self._get_data(data)
        self.estimator = self._get_estimator(estimator + ".pkl")

    def _get_data(self, data):
        try:
            df = pd.read_csv(config[data], sep="\t")
        except Exception as e:
            self.logger.error(e)
            exit()
        else:
            return df

    def _get_estimator(self, estimator_name):
        try:
            estimator = joblib.load(config["model_path"] / estimator_name)
        except Exception as e:
            self.logger.error(e)
            exit()
        else:
            return estimator
