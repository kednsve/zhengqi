import joblib
import pandas as pd

from config import config
from utils import Logger


class Steam:
    def __init__(self, data, log_name, estimator=None):
        self.logger = self._get_logger(log_name)
        self.data = self._get_data(data)
        self.estimator = self._get_estimator(estimator)

    @staticmethod
    def _get_logger(log_name):
        logger = Logger(log_name).get_logger()
        logger.info("logger online")
        return logger

    def _get_data(self, data):
        try:
            df = pd.read_csv(config[data], sep="\t")
            self.logger.info("data load " + data)
        except Exception as e:
            self.logger.error("data loading error")
            self.logger.error("=" * 30)
            self.logger.error(e)
            self.logger.error("=" * 30)
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
            self.logger.error("=" * 30)
            self.logger.error(e)
            self.logger.error("=" * 30)
            exit()
        else:
            return estimator
