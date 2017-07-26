import matplotlib.pyplot as plt
import pandas as pd
from minepy import MINE


def rank(X, y):
    """
    Rank each feature using mutual information
    of target variable 'y' and the feature.
    This is done via Maximal information coefficient.
    This function is linear in number of features.

    :param X: data drame with features
    :param y: target variable
    :return: data frame with rank for each feature name
    ordered decreasing by rank
    """
    ranks = []
    for n, col in enumerate(X.columns):
        m = MINE()
        m.compute_score(y, X[col])
        ranks.append(m.mic())

    ranked = pd.DataFrame()
    ranked['name'] = list(X.columns)
    ranked['rank'] = ranks
    ranked.sort_values('rank', inplace=True, ascending=False)

    return ranked


def plot_distributions(X, y, ranked, nrows=10):
    """
    Plots ordered by rank feature distributions.
    Distribution of each class in 'y' is plotted separatly.

    :param X: data drame with features
    :param y: target variable
    :param ranked: feature ranking result
    """
    fig, axes = plt.subplots(nrows=nrows)

    for n, col in enumerate(ranked['name'][:nrows]):
        sub = X[col]
        sub[y == 1].plot.density(ax=axes[n])
        sub[y != 1].plot.density(ax=axes[n])
        axes[n].set_ylabel(col)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0)


if __name__ == '__main__':
    fname = '/home/guyos/Documents/task/task_data.csv.gz'

    df = pd.read_csv(fname, index_col=0)
    X, y = df.drop('class_label', axis=1), df['class_label']
    ranked = rank(X, y)
    plot_distributions(X, y, ranked)

    # df.plot.scatter('sensor6', 'class_label',
    #                 c='class_label', cmap=plt.get_cmap('coolwarm'))
