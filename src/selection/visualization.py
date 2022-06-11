import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, Union, List
from src.utils import get_hexcolor_by_value
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
plt.style.use('ggplot')


def plot_timeseries_cv(ts_splitter: TimeSeriesSplit, data: pd.DataFrame,
                       metrics: Dict[str, Union[float, List[float]]] = None) -> None:
    """Plots the RMSE for each split of a given timeseries splitter. The color of each train/test bar corresponds to the
    achieved RMSE with green indicating lower value and red indicating higher values. If metrics==None, a schematic
    representation of the timeseries splitter is plotted."""
    plt.figure(figsize=(10, 5))

    split_meta: Dict[str, dict] = {}
    split_count = 0
    for train_index, test_index in ts_splitter.split(data):
        split_count += 1
        train_index, test_index = data.iloc[train_index].index, data.iloc[test_index].index
        split_meta[f"split_{split_count}_train"] = {'start': train_index.min(), 'end': train_index.max()}
        split_meta[f"split_{split_count}_test"] = {'start': test_index.min(), 'end': test_index.max()}

    for split, info in split_meta.items():
        split_num = int(split[split.find("_") + 1: split.rfind("_")])
        y_start = (split_num - 1) * 2 + 1
        y_end = y_start + 1

        date_range = pd.date_range(info['start'], info['end'])
        is_train = split[-5:] == 'train'
        if metrics is None:
            color = '#a3a3a3' if is_train else '#707070'
            alpha = 0.9
        else:
            color = get_hexcolor_by_value(metrics['splits_rmse'][split_num-1], metrics['splits_rmse'], reverse=False)
            alpha = 0.5 if is_train else 0.9
        plt.fill_between(date_range, y_start, y_end, color=color, alpha=alpha)
        if is_train and metrics is not None:
            text_label = f"Acc {round(metrics['splits_rmse'][split_num-1], 4)}"
            plt.text(x=info['start'], y=y_start + .2, s=text_label, fontsize=9, color='black', fontweight='bold')
    plt.gca().get_yaxis().set_visible(False)
    if metrics is not None:
        plt.title(f"Average Accuracy: {round(metrics['avg_rmse'], 4)}")
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.append(mpatches.Patch(color='#a3a3a3', label='Train'))
    handles.append(mpatches.Patch(color='#707070', label='Test'))
    plt.legend(handles=handles, loc="lower right")


def plot_estimator_comparison(scores: Dict[str, float]):
    fig, ax = plt.subplots(figsize=(8.6, len(scores.keys())/1.3))
    ax.set_title("Comparison of Estimators", fontdict={'fontsize': 11.5, 'fontweight': 'bold'})
    ax.set_xlabel('Accuracy')
    scores_series = pd.Series(scores).sort_values()
    h_bars = ax.barh(scores_series.index, scores_series.values)
    ax.bar_label(h_bars, fmt='%.2f')


def plot_bet_return_comparison(estimator_thresholds: Dict[str, List[float]],
                               estimator_returns: Dict[str, List[List[float]]], kind: str = 'both'):
    kinds_accepted = ['mean', 'sum', 'both']
    kind = kind.lower()
    if kind not in kinds_accepted:
        raise ValueError(f"kind '{kind}' not supported. Please use one of {', '.join(kinds_accepted)}.")
    if kind == 'both':
        fig, ax = plt.subplots(2, figsize=(10, 7))
        for estimator, returns in estimator_returns.items():
            ax[0].plot(estimator_thresholds[estimator], [np.sum(r) for r in returns], label=estimator)
            ax[1].plot(estimator_thresholds[estimator], [np.mean(r) for r in returns], label=estimator)
        ax[0].axhline(0, ls='--', lw=.5, color='green')
        ax[1].axhline(0, ls='--', lw=.5, color='green')
        ax[0].set_title("Total vs. Average Betting Return", fontdict={'fontsize': 11.5, 'fontweight': 'bold'})
        ax[0].set_ylabel('Total Return')
        ax[1].set_ylabel('Avg. Return')
        ax[1].set_xlabel('Threshold')
        ax[0].legend()


