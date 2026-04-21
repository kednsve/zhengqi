import numpy as np

from config import config
from .steam import Steam


def data_pre(steam: Steam):
    steam.logger.info("get best model")
    steam.set_best_model()
    steam.logger.info("predict")
    y_pre = steam.estimator.predict(steam.data)
    steam.logger.info(f"y_pre={y_pre}")
    res = np.array(y_pre).reshape(-1, 1)
    np.savetxt(config["data_path"] / "zhengqi_test_pre.txt", res, fmt="%.3f")
    steam.logger.info("save result")
