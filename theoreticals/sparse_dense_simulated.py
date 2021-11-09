import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA

ndp = 10000

np.random.seed(0)
data = np.random.rand(ndp, 2)
x = data[:, 0]
y = data[:, 1]

cond1 = y > 0.5
cond2 = x < 0.1
cond3 = np.random.rand(ndp) < .05
cond4 = np.random.rand(ndp) > 0.9995

data_low = data[(cond1 & cond2) | cond4]

data_high = data[cond3]
print(sum(cond1 & cond2), sum(cond3))


def avg_4dist(dm):
    return np.mean(np.partition(squareform(pdist(dm)), kth=4, axis=1)[:, 4])


print(avg_4dist(data_low), avg_4dist(data_high), avg_4dist(data_high) / avg_4dist(data_low))
del data


def do(d, ax):
    # Plot data

    ax.plot(d[:, 0], d[:, 1], 'b.')
    assert isinstance(ax, plt.Axes)
    ax.set_aspect('equal')

    # Add PCA arrow
    z = PCA(n_components=1, random_state=1)
    z.fit(d)
    pca = z.components_[0]
    if d is data_low:
        pca = -pca

    ex_var = z.explained_variance_ratio_[0]
    ax.set_xlabel(f'$s_E$ = {avg_4dist(d) / (2 ** 0.5):.3f}, ev={ex_var:.2f}', fontsize=12)
    ax.arrow(0, 0, pca[0], pca[1], color='r', head_width=0.05)
    ax.plot([0, 1, 1], [1, 0, 1], 'k.', ms=15)
    # ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticklabels(map(str, [0, 1]), fontsize=12)
    ax.set_yticklabels(map(str, [0, 1]), fontsize=12)


f, a = plt.subplots(1, 2)
do(data_low, a[0])
do(data_high, a[1])
plt.savefig('paper_results/sparsity.eps', bbox_inches='tight')
plt.show()
