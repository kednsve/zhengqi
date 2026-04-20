from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from src.do_gridsearch import do_gridsearch


def gbr_model(x_train, y_train, x_test, y_test,logger):
    logger.info("try GradientBoostingRegressor")
    gb = GradientBoostingRegressor()
    params = {
        "n_estimators": [125, 150, 175],
        "max_depth": [15, 20, 25],
        "learning_rate": [0.1, 0.07, 0.15],
    }
    gb: GradientBoostingRegressor = do_gridsearch(
        gb, params, x_train, y_train, logger
    )
    logger.info("predict")
    y_pre = gb.predict(x_test)
    logger.info(f"MSE: {mean_squared_error(y_test, y_pre)}")
