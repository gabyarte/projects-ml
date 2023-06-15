import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA


def plot_features_scores(
    X, scores, method='univariate', metric_label=r'$-log(p_{value})$'):
    scores_df = pd.DataFrame({'scores': scores, 'columns': X.columns}) \
        .sort_values('scores', ascending=False)
    plt.figure(figsize=(10, 20))
    sns.barplot(data=scores_df, x='scores', y='columns', color='blue')
    plt.grid(True)
    plt.title(f'Feature {method} score')
    plt.xlabel(f'{method.capitalize()} score ({metric_label})')
    plt.ylabel('')
    plt.show()


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
