from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from .GBR_model import gbr_model
from .do_gridsearch import do_gridsearch
from .steam import Steam


def train(steam: Steam):
    steam.logger.info("split train and test")
    x_train, x_test, y_train, y_test = train_test_split(
        steam.data.iloc[:, :-1], steam.data.iloc[:, -1], test_size=0.2, random_state=22
    )
    steam.logger.info("init estimator -- RandomForestRegressor")
    if "RandomForestRegressor" not in steam.temp["models"]:
        steam.estimator = RandomForestRegressor(random_state=22, n_jobs=-1)
        params = {"n_estimators": [125, 150, 175], "max_depth": [15, 20, 25]}
        steam.logger.info("get best estimator")
        steam.estimator = do_gridsearch(
            steam.estimator, params, x_train, y_train, steam.logger
        )
        # max_depth = 20  n_estimators = 175
        steam.logger.info("predict")
        y_pre = steam.estimator.predict(x_test)
        mse = mean_squared_error(y_test, y_pre)
        # MSE: 0.14280621052608308
        steam.logger.info(f"MSE: {mse}")
        steam.add_model(steam.estimator, mse)
    else:
        steam.logger.info("RandomForestRegressor")
        steam.logger.info(f"|-- MSE: {steam.temp['MSE']['RandomForestRegressor']}")
        steam.logger.info("----------------------------------------------------")
    if "GradientBoostingRegressor" not in steam.temp["models"]:
        steam.estimator = gbr_model(x_train, y_train, steam.logger)
        y_pre = steam.estimator.predict(x_test)
        mse = mean_squared_error(y_test, y_pre)
        # MSE: 0.14280621052608308
        steam.logger.info(f"MSE: {mse}")
        steam.add_model(steam.estimator, mse)
    # gbr_model(x_train, y_train,x_test, y_test, steam.logger)
    # MSE: 0.23262583663881684
