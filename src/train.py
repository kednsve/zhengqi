from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .do_gridsearch import do_gridsearch
from .steam import Steam


def train(steam: Steam):
    steam.logger.info("split train and test")
    x_train, x_test, y_train, y_test = train_test_split(
        steam.data.iloc[:, :-1], steam.data.iloc[:, -1], test_size=0.2, random_state=22
    )
    steam.logger.info("init estimator -- RandomForestRegressor")
    steam.estimator = RandomForestRegressor(random_state=22, n_jobs=-1)
    params = {"n_estimators": [125, 150, 175], "max_depth": [15, 20, 25]}
    steam.logger.info("get best estimator")
    steam.estimator = do_gridsearch(
        steam.estimator, params, x_train, y_train, steam.logger
    )
    # max_depth = 20  n_estimators = 175
    steam.logger.info("predict")
    y_pre = steam.estimator.predict(x_test)
    steam.logger.info(f"MSE: {mean_squared_error(y_test, y_pre)}")
    steam.logger.info("try GradientBoostingRegressor")
    gb = GradientBoostingRegressor()
    params = {
        "n_estimators": [125, 150, 175],
        "max_depth": [15, 20, 25],
        "learning_rate": [0.1, 0.07, 0.15],
    }
    gb: GradientBoostingRegressor = do_gridsearch(
        gb, params, x_train, y_train, steam.logger
    )
    steam.logger.info("predict")
    y_pre = gb.predict(x_test)
    steam.logger.info(f"MSE: {mean_squared_error(y_test, y_pre)}")
