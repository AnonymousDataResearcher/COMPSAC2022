import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist

import strings
from data.summary import get_summary
from metric import Metric, WEIGHTED_JACCARD_VANILLA, MANHATTAN_VANILLA, EUCLIDEAN_VANILLA, ALL_VANILLA_METRICS
from metric_handler import MetricHandler


def simulated_plot_thing():
    N_DATAPOINTS = 1000

    def get(n, seed, mode):
        rng = np.random.RandomState(seed)
        data = rng.rand(N_DATAPOINTS, n)
        return pdist(data, mode).mean()

    def do(n, seeds, mode):
        return np.array([get(n, s, mode) for s in seeds]).mean()

    def plot(mode):
        f, ax = plt.subplots()
        x = np.arange(4, 65, 4)
        y = [do(xi, range(10), mode=mode) for xi in x]
        ax.plot(x, y, 'r.', ms=10, label='simulated')
        ax.set_xlabel('$n$')
        ax.set_ylabel('E[dist]')

        ls = np.linspace(0, 64, 1000)
        if mode == 'euclidean':
            ab, r, *_ = np.polyfit(np.sqrt(x), y, 1, full=True)
            lab = f'{ab[0]:.2f}\u221Ax+{ab[1]:.2f}'
            y2 = ab[0] * np.sqrt(ls) + ab[1]
        else:
            ab, r, *_ = np.polyfit(x, y, 1, full=True)
            lab = f'{ab[0]:.2f}x+{ab[1]:.2f}'
            y2 = ab[0] * ls + ab[1]
        ax.plot(ls, y2, 'k:', label=lab + f', r={r[0]:.2e}')
        ax.legend()
        ax.set_title(mode)
        plt.show()

    plot('euclidean')
    plot('cityblock')


def average_distance_thing():
    df = get_summary()
    df = df[df[strings.NUMBER_OF_DATAPOINTS] < 100000]
    res = pd.DataFrame(index=df.index, columns=ALL_VANILLA_METRICS)

    for dsn in df.index:
        for m in ALL_VANILLA_METRICS:
            print(dsn, m, sep='\t')
            c = MetricHandler(m).clusterable_from_raw(dsn)
            s = 0
            for i in range(c.size):
                s += c.get_sub_distance_matrix([i], None).sum()
            res.loc[dsn, m] = s / df.loc[dsn, strings.NUMBER_OF_DATAPOINTS] ** 2

        res.to_csv('results/test.csv')


if __name__ == '__main__':
    average_distance_thing()
