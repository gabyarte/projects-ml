import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA


def plot_features_scores(
    X, scores, method='univariate', metric_label=r'$-log(p_{value})$',
    figsize=(10, 20), ax=None
):
    scores_df = pd.DataFrame({'scores': scores, 'columns': X.columns}) \
        .sort_values('scores', ascending=False)
    if ax is None:
        plt.figure(figsize=figsize)
    sns.barplot(data=scores_df, x='scores', y='columns', color='blue', ax=ax)
    (ax or plt).grid(True)

    set_title, set_x, set_y = (plt.title, plt.xlabel, plt.ylabel) \
        if ax is None \
        else (ax.set_title, ax.set_xlabel, ax.set_ylabel)

    set_title(f'Feature {method} score')
    set_x(f'{method.capitalize()} score ({metric_label})')
    set_y('')


def decision_boundary_plot(estimator, X, y, model='model'):
    _X = PCA(n_components=2).fit_transform(X, y)
    _X = pd.DataFrame(_X)

    decision_boundary = DecisionBoundaryDisplay.from_estimator(
        estimator.fit(_X, y['y']),
        _X,
        response_method='predict',
        cmap=plt.cm.coolwarm,
        alpha=0.8
    )
    decision_boundary.ax_.scatter(
        x=_X[0],
        y=_X[1],
        c=y['y'],
        cmap=plt.cm.coolwarm,
        s=20,
        edgecolors='k'
    )
    plt.title(f'Decision boundary of {model}')
    plt.show()


def plot_resampling(X, y, sampler):
    fig, axs = plt.subplots(1, 2)

    _X = PCA(n_components=2).fit_transform(X, y)
    _X = pd.DataFrame(_X)

    axs[0].scatter(
        _X[0],
        _X[1],
        c=y['y'],
        alpha=0.8,
        edgecolor="k"
    )
    axs[0].set_title('Before resampling')
    sns.despine(ax=axs[0], offset=10)

    X_res, y_res = sampler.fit_resample(X, y)
    _X_res = PCA(n_components=2).fit_transform(X_res, y_res)
    _X_res = pd.DataFrame(_X_res)

    axs[1].scatter(
        _X_res[0],
        _X_res[1],
        c=y_res['y'],
        alpha=0.8,
        edgecolor="k"
    )
    axs[1].set_title('After resampling')

    sns.despine(ax=axs[1], offset=10)

    return fig
