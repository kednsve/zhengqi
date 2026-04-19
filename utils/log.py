import logging

from config import config


class Logger:
    def __init__(self, log_name):
        self.log_name = log_name + ".log"
        self.fmt = config["log_fmt"]
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(logging.INFO)

    def get_logger(self):
        log_path = config["root_path"] / "logs"
        log_path.mkdir(parents=True, exist_ok=True)
        rotate = logging.FileHandler(log_path / self.log_name, mode="a")
        rotate.setFormatter(logging.Formatter(self.fmt))
        self.logger.addHandler(rotate)
        return self.logger
