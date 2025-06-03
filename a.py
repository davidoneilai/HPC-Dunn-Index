import multiprocessing as mp
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.datasets import load_iris
from sklearn.metrics import pairwise_distances


def compute_delta_pair(args):
    X, labels, i, j = args
    cluster_i = X[labels == i]
    cluster_j = X[labels == j]
    return np.min(pairwise_distances(cluster_i, cluster_j))


if __name__ == "__main__":

    def dunn_index_serial(X, labels):
        distances = squareform(pdist(X))
        unique_labels = np.unique(labels)
        deltas = []
        diameters = []

        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                cluster_i = X[labels == unique_labels[i]]
                cluster_j = X[labels == unique_labels[j]]
                delta = np.min(pairwise_distances(cluster_i, cluster_j))
                deltas.append(delta)

        for k in unique_labels:
            cluster_k = X[labels == k]
            diameter = np.max(pdist(cluster_k)) if len(cluster_k) > 1 else 0
            diameters.append(diameter)

        return np.min(deltas) / np.max(diameters)

    def dunn_index_parallel(X, labels):
        distances = squareform(pdist(X))
        unique_labels = np.unique(labels)
        args_list = [
            (X, labels, unique_labels[i], unique_labels[j])
            for i in range(len(unique_labels))
            for j in range(i + 1, len(unique_labels))
        ]

        with mp.Pool(processes=mp.cpu_count()) as pool:
            deltas = pool.map(compute_delta_pair, args_list)

        diameters = []
        for k in unique_labels:
            cluster_k = X[labels == k]
            diameter = np.max(pdist(cluster_k)) if len(cluster_k) > 1 else 0
            diameters.append(diameter)

        return np.min(deltas) / np.max(diameters)

    print("Carregando dataset Iris...")
    data = load_iris()
    X = data.data
    labels = data.target

    times_serial = []
    times_parallel = []
    valid_sizes = []
    sizes = list(range(10, 151, 10))

    print("Iniciando testes de desempenho...\n")
    for size in sizes:
        X_subset = X[:size]
        labels_subset = labels[:size]

        if len(np.unique(labels_subset)) < 2:
            continue

        valid_sizes.append(size)
        print(f"Tamanho: {size} amostras")

        start = time.time()
        dunn_index_serial(X_subset, labels_subset)
        t_serial = time.time() - start
        times_serial.append(t_serial)
        print(f"Tempo serial: {t_serial:.6f} segundos")

        start = time.time()
        dunn_index_parallel(X_subset, labels_subset)
        t_parallel = time.time() - start
        times_parallel.append(t_parallel)
        print(f"Tempo paralelo: {t_parallel:.6f} segundos\n")

    print("Testes concluídos!\n")
    print("Resultados resumidos:")
    for i in range(len(valid_sizes)):
        print(
            f"- {valid_sizes[i]} amostras: Serial={times_serial[i]:.6f}s | Paralelo={times_parallel[i]:.6f}s"
        )

    plt.figure(figsize=(10, 6))
    plt.plot(valid_sizes, times_serial, label="Serial", marker="o")
    plt.plot(
        valid_sizes, times_parallel, label="Paralelo (Multiprocessing)", marker="x"
    )
    plt.xlabel("Tamanho do dataset (n amostras)")
    plt.ylabel("Tempo de execução (s)")
    plt.title("Comparação de desempenho: Serial vs Paralelo (Índice de Dunn)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_dunn_comparacao.png")
    plt.show()

    input("\n🖼️ Pressione Enter para sair...")
