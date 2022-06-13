from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_validate, cross_val_predict
from sklearn.metrics import accuracy_score
from src.selection.visualization import plot_timeseries_cv, plot_estimator_comparison, plot_bet_return_comparison
from typing import List, Union, Dict

_ts_splitter = TimeSeriesSplit(n_splits=7, test_size=70, max_train_size=400)


def create_train_test(df_in: pd.DataFrame, split_date: Union[str, datetime], start_date: Union[str, datetime] = None,
                      features: List[str] = None, leagues: List[str] = None):
    df = df_in.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')

    if start_date is not None:
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        df = df[df['date'] > start_date]
    if leagues is not None:
        df = df[df['league_id'].isin(leagues)]
    if not isinstance(split_date, datetime):
        split_date = pd.to_datetime(split_date)
    train_df = df[df['date'] < split_date]
    test_df = df[df['date'] >= split_date]
    if features is not None:
        train_df, test_df = train_df[features + ['result']], test_df[features + ['result']]
    train_df.dropna(inplace=True)
    test_df.dropna(inplace=True)
    rows_dropped = len(df) - (len(train_df) + len(test_df))
    if rows_dropped > 0:
        print(f"Warning: {rows_dropped} rows dropped because of NA values.")
    X_train, y_train = train_df.drop(columns=['result']), train_df['result']
    X_test, y_test = test_df.drop(columns=['result']), test_df['result']
    return X_train, X_test, y_train, y_test


def temp_cross_validate(model, X, y, visualize: bool = False, dates: pd.Series = None) -> float:
    cv = cross_validate(model, X, y, scoring=['accuracy'], cv=_ts_splitter)
    avg_accuracy = cv['test_accuracy'].mean()
    if visualize:
        metrics = {'avg_rmse': avg_accuracy, 'splits_rmse': cv['test_accuracy']}
        if dates is not None:
            X_plot = X.set_index(pd.to_datetime(dates))
        plot_timeseries_cv(_ts_splitter, X_plot, metrics)
    return avg_accuracy


def compare_estimator_accuracy(estimators: list, X: pd.DataFrame, y: pd.Series, bookmaker_pred: pd.Series = None) -> None:
    scores: Dict[str, float] = {}
    for estimator in estimators:
        estimator_name: str = estimator.__class__.__name__
        estimator_score = temp_cross_validate(estimator, X, y)
        scores[estimator_name] = estimator_score
    test_indices = []
    for train_index, test_index in _ts_splitter.split(X):
        y_test = y.iloc[test_index]
        test_indices += list(y_test.index)
    if bookmaker_pred is not None:
        scores['Bookmaker'] = accuracy_score(y.loc[test_indices], bookmaker_pred.loc[test_indices])
    scores['Always Home'] = y.loc[test_indices].value_counts(normalize=True).max()
    plot_estimator_comparison(scores)


def compare_betting_return(estimators: list, X: pd.DataFrame, y: pd.Series, match_df: pd.DataFrame) -> None:
    estimator_returns: Dict[str, List[List[float]]] = {}
    estimator_thresholds: Dict[str, List[float]] = {}
    for estimator in estimators:
        test_indices = []
        y_pred_proba = None
        for train_index, test_index in _ts_splitter.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            estimator.fit(X_train, y_train)
            y_pred_proba_split = estimator.predict_proba(X_test)
            if y_pred_proba is None:
                y_pred_proba: np.ndarray = y_pred_proba_split
            else:
                y_pred_proba = np.vstack((y_pred_proba, y_pred_proba_split))
            test_indices += list(X_test.index)

        y_pred_proba_df = pd.DataFrame(y_pred_proba, columns=['A', 'D', 'H'], index=test_indices)
        y_pred_series = y_pred_proba_df.idxmax(axis=1)
        y_pred_proba_adj = pd.DataFrame(index=y_pred_proba_df.index)
        y_pred_proba_adj['Pred'] = y_pred_series
        y_pred_proba_adj['PredProba'] = y_pred_proba_df.max(axis=1)
        y_pred_proba_adj['BmProba'] = 1 / match_df.loc[test_indices].apply(
            lambda r: r['b365_H'] if y_pred_series.loc[r.name] == 'H'
            else r['b365_D'] if y_pred_series.loc[r.name] == 'D'
            else r['b365_A'], axis=1)

        pred_proba_dif = y_pred_proba_adj['PredProba'] - y_pred_proba_adj['BmProba']
        biggest_dif = pred_proba_dif.max()
        smallest_dif = pred_proba_dif.min()
        tested_thresholds = [t/100 for t in range(int(smallest_dif*100)-1, int(biggest_dif*100)+1)]
        returns = []
        for threshold in tested_thresholds:
            y_pred_confident = y_pred_proba_adj[pred_proba_dif > threshold]['Pred']

            bet_correct: pd.Series = (y.loc[y_pred_confident.index] == y_pred_confident).astype(int)
            odds: pd.Series = match_df.loc[y_pred_confident.index].apply(
                lambda r: r['b365_H'] if y_pred_confident.loc[r.name] == 'H' else r['b365_D'] if y_pred_confident.loc[
                                                                                                   r.name] == 'D' else r[
                    'b365_A'], axis=1)
            return_amt = bet_correct * odds - 1
            returns.append(return_amt)
        estimator_returns[estimator.__class__.__name__] = returns
        estimator_thresholds[estimator.__class__.__name__] = tested_thresholds
    plot_bet_return_comparison(estimator_thresholds, estimator_returns)


def compare_estimators(estimators: list, X: pd.DataFrame, y: pd.Series, match_df: pd.DataFrame,
                       bookmaker_pred: pd.Series = None) -> None:
    compare_estimator_accuracy(estimators, X, y, bookmaker_pred)
    compare_betting_return(estimators, X, y, match_df)