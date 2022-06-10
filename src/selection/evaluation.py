from sklearn.model_selection import TimeSeriesSplit, cross_validate


def create_train_test(df):
    pass

def temp_cross_validate(model, X, y):
    ts_splitter = TimeSeriesSplit(n_splits=5, test_size=100, max_train_size=500)
    cv = cross_validate(model, X, y, scoring=['accuracy'], cv=ts_splitter)
    avg_accuracy = cv['test_accuracy'].mean()
    return avg_accuracy
