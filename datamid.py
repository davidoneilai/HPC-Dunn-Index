import pandas as pd
import numpy as np
import time
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def dunn_index_serial(X, labels):
    print("🔁 [Serial] Iniciando cálculo...")
    start_time = time.time()

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

    dunn = np.min(deltas) / np.max(diameters)
    print(f"✅ [Serial] Dunn index: {dunn:.4f} | Tempo: {time.time() - start_time:.2f}s\n")
    return dunn

def compute_delta_pair(args):
    X, labels, i, j = args
    cluster_i = X[labels == i]
    cluster_j = X[labels == j]
    return np.min(pairwise_distances(cluster_i, cluster_j))

def dunn_index_parallel(X, labels):
    print("🔁 [Parallel] Iniciando cálculo...")
    start_time = time.time()

    unique_labels = np.unique(labels)
    args_list = [(X, labels, unique_labels[i], unique_labels[j])
                 for i in range(len(unique_labels))
                 for j in range(i + 1, len(unique_labels))]

    with mp.Pool(mp.cpu_count()) as pool:
        deltas = pool.map(compute_delta_pair, args_list)

    diameters = []
    for k in unique_labels:
        cluster_k = X[labels == k]
        diameter = np.max(pdist(cluster_k)) if len(cluster_k) > 1 else 0
        diameters.append(diameter)

    dunn = np.min(deltas) / np.max(diameters)
    print(f"✅ [Parallel] Dunn index: {dunn:.4f} | Tempo: {time.time() - start_time:.2f}s\n")
    return dunn

def processar_dataset(path_excel, n_max=300000):
    print(f"📥 Carregando '{path_excel}' com até {n_max} linhas...")
    df = pd.read_excel(path_excel)
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    data_ref = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg({
        "InvoiceDate": lambda x: (data_ref - x.max()).days,
        "InvoiceNo": "nunique",
        "TotalPrice": "sum"
    })
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm.reset_index(inplace=True)

    if len(rfm) > n_max:
        rfm = rfm.sample(n=n_max, random_state=42)

    print(f"✅ Dataset final: {len(rfm)} clientes")
    return rfm

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  

    rfm = processar_dataset("datamid.xlsx", n_max=300000)

    X = StandardScaler().fit_transform(rfm[["Recency", "Frequency", "Monetary"]])
    kmeans = KMeans(n_clusters=4, random_state=42)
    labels = kmeans.fit_predict(X)
    
    print("\n⏱ Medindo tempo de execução:")

    start = time.time()
    dunn_s = dunn_index_serial(X, labels)
    tempo_s = time.time() - start

    start = time.time()
    dunn_p = dunn_index_parallel(X, labels)
    tempo_p = time.time() - start

    plt.figure(figsize=(6, 4))
    plt.bar(["Serial", "Paralelo"], [tempo_s, tempo_p], color=["skyblue", "orange"])
    plt.ylabel("Tempo (s)")
    plt.title("Tempo de Execução do Índice de Dunn")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("grafico_tempo_execucao.png")
    plt.show()

    plt.figure(figsize=(6, 4))
    unique, counts = np.unique(labels, return_counts=True)
    plt.bar(unique, counts, color="green")
    plt.xlabel("Cluster")
    plt.ylabel("Número de Clientes")
    plt.title("Distribuição de Clientes por Cluster")
    plt.tight_layout()
    plt.savefig("grafico_distribuicao_clusters.png")
    plt.show()

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(6, 5))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=10)
    plt.title("Visualização dos Clusters via PCA")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_pca_clusters.png")
    plt.show()

    rfm["Cluster"] = labels
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for idx, col in enumerate(["Recency", "Frequency", "Monetary"]):
        axs[idx].set_title(col)
        rfm.boxplot(column=col, by="Cluster", ax=axs[idx])
        axs[idx].set_xlabel("Cluster")
        axs[idx].set_ylabel(col)

    plt.suptitle("Distribuição das Métricas RFM por Cluster")
    plt.tight_layout()
    plt.savefig("grafico_boxplot_rfm.png")
    plt.show()