import numpy as np
import matplotlib.pyplot as plt

def plot_features_scores(
    X, scores, y_label=r'Univariate score ($-Log(p_{value})$)'):
    X_indices = np.arange(X.shape[-1])
    plt.figure(1)
    plt.clf()
    plt.bar(X_indices - 0.05, scores, width=0.2)
    plt.title("Feature univariate score")
    plt.xlabel("Feature number")
    plt.ylabel(y_label)
    plt.show()
