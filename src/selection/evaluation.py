from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score
from src.selection.visualization import plot_timeseries_cv, plot_estimator_comparison, plot_bet_return_comparison
from typing import List, Union, Dict
from src.features.pipeline import construct_full_pipeline


# Time series splitter that is reused throughout the validation
#_ts_splitter = TimeSeriesSplit(n_splits=7, test_size=500, max_train_size=1500)
_ts_splitter = TimeSeriesSplit(n_splits=7, test_size=1500)#, max_train_size=6000)


def create_train_test(df_in: pd.DataFrame, split_date: Union[str, datetime], start_date: Union[str, datetime] = None,
                      features: List[str] = None, leagues: List[str] = None):
    """Splits a given dataframe into train and test around a specified split date. The target variable passed must be
    named 'result'."""
    df = df_in.copy()
    if 'date' not in df.columns:
        df['date'] = df.index
        df.index.name = None
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    # Filter df to only contain values rows after start date (if start_date is specified)
    if start_date is not None:
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        df = df[df['date'] > start_date]

    # Filter for leagues (if leagues specified)
    if leagues is not None:
        df = df[df['league_id'].isin(leagues)]

    # Split into train and test around split date
    if not isinstance(split_date, datetime):  # cast split_date to datetime object
        split_date = pd.to_datetime(split_date)
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]

    # Filter for features (if features specified)
    if features is not None:
        train_df, test_df = train_df[features + ['result']], test_df[features + ['result']]

    # Drop NAs and issue warning if NAs were dropped
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    rows_dropped = len(df) - (len(train_df) + len(test_df))
    if rows_dropped > 0:
        print(f"Warning: {rows_dropped} rows dropped because of NA values.")

    # Split into train and test
    X_train, y_train = train_df.drop(columns=['result']), train_df['result']
    X_test, y_test = test_df.drop(columns=['result']), test_df['result']
    return X_train, X_test, y_train, y_test


def temp_cross_validate(estimator, X, y, visualize: bool = False, dates: pd.Series = None) -> float:
    """Cross validates a given model over several time series splits and returns the average accuracy over all splits.
    """
    full_pipeline = construct_full_pipeline(estimator)
    cv = cross_validate(full_pipeline, X, y, scoring=['accuracy'], cv=_ts_splitter, n_jobs=6)
    avg_accuracy = cv['test_accuracy'].mean()
    if visualize:
        metrics = {'avg_rmse': avg_accuracy, 'splits_rmse': cv['test_accuracy']}
        if dates is not None:  # index needs to be the date of each row/match
            X_plot = X.set_index(pd.to_datetime(dates))
        plot_timeseries_cv(_ts_splitter, X_plot, metrics)
    return avg_accuracy


def compare_estimator_accuracy(estimators: list, X: pd.DataFrame, y: pd.Series, bookmaker_pred: pd.Series = None):
    """Compares the average accuracy of several estimators over multiple time splits. Plots the results as horizontal
    bar chart, including the accuracy of the bookmaker and, as a baseline, the potential accuracy if all bets were set
    on the home team."""
    # Get the average accuracy for each estimator
    scores: Dict[str, float] = {}  # Dict to store the accuracy for each estimator in
    for estimator in estimators:
        estimator_name: str = estimator.__class__.__name__
        estimator_score = temp_cross_validate(estimator, X, y)
        scores[estimator_name] = estimator_score

    # Get the indices of used test sets (for comparison with bookmaker and baseline)
    test_indices = []
    for train_index, test_index in _ts_splitter.split(X):
        y_test = y.iloc[test_index]
        test_indices += list(y_test.index)

    # Get the accuracy of bookmaker and always hoe prediction
    if bookmaker_pred is not None:
        scores['Bookmaker'] = accuracy_score(y.loc[test_indices], bookmaker_pred.loc[test_indices])
    scores['Always Home'] = y.loc[test_indices].value_counts(normalize=True).max()

    plot_estimator_comparison(scores)


def compare_betting_return(estimators: list, X: pd.DataFrame, y: pd.Series, match_df: pd.DataFrame) -> None:
    """Compares the returns of betting at different thresholds of the predicted probability with that of the bookmaker
    probability for different estimators. Plots the results as total and average for each threshold."""
    estimator_returns: Dict[str, List[List[float]]] = {}  # Dict to store the returns per threshold per estimator
    estimator_thresholds: Dict[str, List[float]] = {}  # Dict to store the used thresholds per estimator
    for estimator in estimators:
        test_indices = []  # the indices of matches used as test sets
        y_pred_proba: Union[np.ndarray, None] = None  # None type object that will be overridden with a Numpy array

        # Manually perform time series cross validation
        for train_index, test_index in _ts_splitter.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            full_pipeline = construct_full_pipeline(estimator)
            full_pipeline.fit(X_train, y_train)
            y_pred_proba_split = full_pipeline.predict_proba(X_test)

            # Save predicted probabilities
            if y_pred_proba is None:
                y_pred_proba = y_pred_proba_split
            else:
                y_pred_proba = np.vstack((y_pred_proba, y_pred_proba_split))

            # Save used test indices
            test_indices += list(X_test.index)

        y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=['A', 'D', 'H'], index=test_indices)
        y_pred = y_pred_proba_df.idxmax(axis=1)  # get the 'absolute' prediction
        y_pred_proba_adj = pd.DataFrame(index=y_pred_proba_df.index)
        y_pred_proba_adj['Pred'] = y_pred  # predicted outcome
        y_pred_proba_adj['PredProba'] = y_pred_proba_df.max(axis=1)  # predicted probability of most likely outcome
        y_pred_proba_adj['BmProba'] = 1 / match_df.loc[test_indices].apply(
            lambda r: r['b365_H'] if y_pred.loc[r.name] == 'H'
            else r['b365_D'] if y_pred.loc[r.name] == 'D'
            else r['b365_A'], axis=1)  # Bookmaker prediction

        # Get thresholds as 1% steps between the smallest and greatest difference between bookmaker and estimator proba
        pred_proba_dif = y_pred_proba_adj['PredProba'] - y_pred_proba_adj['BmProba']
        biggest_dif = pred_proba_dif.max()
        smallest_dif = pred_proba_dif.min()
        tested_thresholds = [t/100 for t in range(int(smallest_dif*100)-1, int(biggest_dif*100)+1)]
        returns: List[List[float]] = []  # To store the returns for each threshold
        for threshold in tested_thresholds:
            # Filter predictions for those the estimator is confident on, i.e. that are above the threshold
            y_pred_confident = y_pred_proba_adj[pred_proba_dif > threshold]['Pred']

            # Check if each prediction/bet was correct
            bet_correct: pd.Series = (y.loc[y_pred_confident.index] == y_pred_confident).astype(int)

            # Get the bookmaker odds of the bets
            odds: pd.Series = match_df.loc[y_pred_confident.index].apply(
                lambda r: r['b365_H'] if y_pred_confident.loc[r.name] == 'H'
                else r['b365_D'] if y_pred_confident.loc[r.name] == 'D'
                else r['b365_A'], axis=1)

            # Get the return from each bet
            return_amt: pd.Series = (bet_correct * odds) - 1
            returns.append(return_amt.tolist())

        # Save results and tested thresholds
        estimator_returns[estimator.__class__.__name__] = returns
        estimator_thresholds[estimator.__class__.__name__] = tested_thresholds

    # Plot results
    plot_bet_return_comparison(estimator_thresholds, estimator_returns)


def compare_estimators(estimators: list, X: pd.DataFrame, y: pd.Series, match_df: pd.DataFrame,
                       bookmaker_pred: pd.Series = None) -> None:
    """Compares a set of estimators by comparing their accuracy and the hypothetic return of betting according to
    predidctions."""
    compare_estimator_accuracy(estimators, X, y, bookmaker_pred)
    compare_betting_return(estimators, X, y, match_df)