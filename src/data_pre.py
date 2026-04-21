from .steam import Steam


def data_pre(steam: Steam):
    steam.logger.info("get best model")
    steam.set_best_model()
    steam.logger.info("predict")
    print(steam.estimator)
    # y_pre=steam.estimator.predict(steam.data)
    # print(y_pre)
