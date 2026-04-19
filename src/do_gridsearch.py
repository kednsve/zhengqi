from sklearn.model_selection import GridSearchCV


def do_gridsearch(estimator, params, x_train, y_train, logger, cv=5):
    logger.info("=" * 20 + "GridSearchCV" + "=" * 20)
    logger.info("test params {}".format(params))
    gs = GridSearchCV(estimator, params, n_jobs=-1, cv=cv)
    gs.fit(x_train, y_train)
    logger.info(f"best params: {gs.best_params_}")
    logger.info(f"best score:  {gs.best_score_}")
    logger.info("=" * 20 + "GridSearchCV" + "=" * 20)
    return gs.best_estimator_
