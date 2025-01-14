import numpy as np
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_classification
from sklearn.manifold import TSNE

from lmnn_impl import LMNN


np.random.seed(2024)


def lmnn_fit_transform(X, y, k):
    lmnn = LMNN(k=k) 
    lmnn.fit(X, y)
    X_lmnn = lmnn.transform(X)
    return X_lmnn


def sandwich_demo():
    plt.figure(figsize=(12, 6))
    plt.suptitle("LMNN Sandwich DEMO")

    X, y = sandwich_data()
    knn = nearest_neighbors(X, k=2)

    ax = plt.subplot(1, 2, 1)
    ax.set_xlim([-5, 4])
    ax.set_ylim([-3, 2])
    plot_sandwich_data(X, y, ax)
    plot_neighborhood_graph(X, knn, y, ax)
    ax.set_title("Input Space")
    ax.set_aspect("equal") 

    X_lmnn = lmnn_fit_transform(X, y, 2)
    lmnn_knn = nearest_neighbors(X_lmnn, k=2)

    ax = plt.subplot(1, 2, 2)
    ax.set_xlim([X_lmnn[:, 0].min() - 1, X_lmnn[:, 0].max() + 1])
    ax.set_ylim([X_lmnn[:, 1].min() - 1, X_lmnn[:, 1].max() + 1])
    ax.set_aspect("equal")
    plot_sandwich_data(X_lmnn, y, axis=ax)
    plot_neighborhood_graph(X_lmnn, lmnn_knn, y, axis=ax)
    ax.set_title("LMNN")
    plt.savefig("sand.png")


def tsne_demo():
    plt.figure(figsize=(12, 6))
    plt.suptitle("LMNN T-SNE DEMO")
    X, y = gen_data()
    plt.subplot(1, 2, 1)
    plot_tsne(X, y)
    plt.title("Raw Data")

    X_lmnn = lmnn_fit_transform(X, y, 5)

    plt.subplot(1, 2, 2)
    plot_tsne(X_lmnn, y)
    plt.title("LMNN Transformed Data")
    plt.savefig("a.png")


def gen_data():
    X, y = make_classification(
        n_samples=100,
        n_classes=3,
        n_clusters_per_class=2,
        n_informative=3,
        class_sep=4.0,
        n_features=5,
        n_redundant=0,
        shuffle=True,
        scale=[1, 1, 20, 20, 20],
    )
    return X, y


def plot_tsne(X, y, colors="rbgmky"):
    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    for y_val in np.unique(y):
        X_cls = X_embedded[y == y_val]
        plt.scatter(X_cls[:, 0], X_cls[:, 1], c=colors[y_val], label=f"class {y_val}")
    plt.legend()
    plt.xticks(())
    plt.yticks(())


def nearest_neighbors(X, k=5):
    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    return knn.kneighbors(X, return_distance=False)


def sandwich_data(num_classes=6, num_points=9, dist=0.7):
    """Generate sandwich-like data"""
    data = np.zeros((num_classes, num_points, 2), dtype=float)
    labels = np.zeros((num_classes, num_points), dtype=int)

    x_centers = np.arange(num_points, dtype=float) - num_points / 2
    y_centers = dist * (np.arange(num_classes, dtype=float) - num_classes / 2)
    for i, yc in enumerate(y_centers):
        for k, xc in enumerate(x_centers):
            data[i, k, 0] = np.random.normal(xc, 0.1)
            data[i, k, 1] = np.random.normal(yc, 0.1)
        labels[i, :] = i
    return data.reshape((-1, 2)), labels.ravel()


def plot_sandwich_data(x, y, axis=plt, colors="rbgmky"):
    for idx, val in enumerate(np.unique(y)):
        xi = x[y == val]
        axis.scatter(*xi.T, s=50, facecolors="none", edgecolors=colors[idx])


def plot_neighborhood_graph(x, nn, y, axis=plt, colors="rbgmky"):
    for i, a in enumerate(x):
        b = x[nn[i, 1]]
        axis.plot((a[0], b[0]), (a[1], b[1]), colors[y[i]])


if __name__ == "__main__":
    tsne_demo()
    sandwich_demo()
