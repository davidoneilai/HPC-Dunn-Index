import multiprocessing as mp
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler


def dunn_index_serial(X, labels):
    print("🔁 [Serial] Iniciando cálculo...")
    start_time = time.time()
    print(f"Labels: {np.unique(labels)}")

    unique_labels = np.unique(labels)
    deltas = []
    diameters = []

    for i in range(len(unique_labels)):
        for j in range(i + 1, len(unique_labels)):
            cluster_i = X[labels == unique_labels[i]]
            cluster_j = X[labels == unique_labels[j]]
            print(
                f"Calculando distância mínima entre clusters {unique_labels[i]} e {unique_labels[j]}..."
            )
            delta = np.min(pairwise_distances(cluster_i, cluster_j))

            deltas.append(delta)

    for k in unique_labels:
        print(f"Calculando diâmetro do cluster {k}...")
        cluster_k = X[labels == k]
        diameter = np.max(pdist(cluster_k)) if len(cluster_k) > 1 else 0
        diameters.append(diameter)

    dunn = np.min(deltas) / np.max(diameters)
    print(
        f"✅ [Serial] Dunn index: {dunn:.4f} | Tempo: {time.time() - start_time:.2f}s\n"
    )
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
    args_list = [
        (X, labels, unique_labels[i], unique_labels[j])
        for i in range(len(unique_labels))
        for j in range(i + 1, len(unique_labels))
    ]

    with mp.Pool(mp.cpu_count()) as pool:
        deltas = pool.map(compute_delta_pair, args_list)

    diameters = []
    for k in unique_labels:
        cluster_k = X[labels == k]
        diameter = np.max(pdist(cluster_k)) if len(cluster_k) > 1 else 0
        diameters.append(diameter)

    dunn = np.min(deltas) / np.max(diameters)
    print(
        f"✅ [Parallel] Dunn index: {dunn:.4f} | Tempo: {time.time() - start_time:.2f}s\n"
    )
    return dunn


def processar_dataset(path_csv, n_max=300000):
    print(f"📥 Carregando '{path_csv}' com até {n_max} linhas...")
    df = pd.read_csv(path_csv)
    df.dropna(subset=["CustomerID"], inplace=True)
    df = df[df["Quantity"] > 0]
    df = df[df["UnitPrice"] > 0]

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    data_ref = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = df.groupby("CustomerID").agg(
        {
            "InvoiceDate": lambda x: (data_ref - x.max()).days,
            "InvoiceNo": "nunique",
            "TotalPrice": "sum",
        }
    )
    rfm.columns = ["Recency", "Frequency", "Monetary"]
    rfm.reset_index(inplace=True)

    if len(rfm) > n_max:
        rfm = rfm.sample(n=n_max, random_state=42)

    print(f"✅ Dataset final: {len(rfm)} clientes")
    return rfm


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # Criar pasta para resultados
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    rfm = processar_dataset("data/cleaned_retail.csv", n_max=300000)
    X = StandardScaler().fit_transform(rfm)

    # Armazenar resultados dos experimentos
    resultados_experimentos = []

    print("\n🧪 Iniciando experimentos com diferentes valores de k...")

    # Loop para diferentes valores de k
    for k in range(4, 11):
        print(f"\n{'=' * 50}")
        print(f"🔬 Experimento com k = {k}")
        print(f"{'=' * 50}")

        # K-means clustering
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)

        print(f"\n⏱ Medindo tempo de execução para k = {k}:")

        # Cálculo serial do Dunn index
        start = time.time()
        dunn_s = dunn_index_serial(X, labels)
        tempo_s = time.time() - start

        # Cálculo paralelo do Dunn index
        start = time.time()
        dunn_p = dunn_index_parallel(X, labels)
        tempo_p = time.time() - start

        # Armazenar resultados
        resultado = {
            "k": k,
            "dunn_serial": dunn_s,
            "dunn_parallel": dunn_p,
            "tempo_serial": tempo_s,
            "tempo_parallel": tempo_p,
            "speedup": tempo_s / tempo_p if tempo_p > 0 else 0,
        }
        resultados_experimentos.append(resultado)

        # Criar subpasta para este k
        k_dir = os.path.join(results_dir, f"k_{k}")
        os.makedirs(k_dir, exist_ok=True)

        # Gráfico de tempo de execução
        plt.figure(figsize=(8, 6))
        plt.bar(["Serial", "Paralelo"], [tempo_s, tempo_p], color=["skyblue", "orange"])
        plt.ylabel("Tempo (s)")
        plt.title(f"Tempo de Execução do Índice de Dunn (k = {k})")
        plt.grid(axis="y")
        for i, v in enumerate([tempo_s, tempo_p]):
            plt.text(i, v + 0.1, f"{v:.2f}s", ha="center", va="bottom")
        plt.tight_layout()
        plt.savefig(os.path.join(k_dir, f"tempo_execucao_k_{k}.png"))
        plt.close()

        # Gráfico de distribuição de clusters
        plt.figure(figsize=(8, 6))
        unique, counts = np.unique(labels, return_counts=True)
        plt.bar(unique, counts, color="green")
        plt.xlabel("Cluster")
        plt.ylabel("Número de Clientes")
        plt.title(f"Distribuição de Clientes por Cluster (k = {k})")
        plt.tight_layout()
        plt.savefig(os.path.join(k_dir, f"distribuicao_clusters_k_{k}.png"))
        plt.close()

        # Visualização PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        plt.figure(figsize=(8, 6))
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", s=10)
        plt.title(f"Visualização dos Clusters via PCA (k = {k})")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.colorbar(label="Cluster")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(k_dir, f"pca_clusters_k_{k}.png"))
        plt.close()

        print(f"✅ Experimento k = {k} concluído!")
        print(f"   Dunn Index: {dunn_s:.4f}")
        print(f"   Speedup: {tempo_s / tempo_p:.2f}x")

    # Criar resumo dos experimentos
    df_resultados = pd.DataFrame(resultados_experimentos)
    df_resultados.to_csv(
        os.path.join(results_dir, "resumo_experimentos.csv"), index=False
    )

    # Gráficos comparativos
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Dunn Index vs k
    ax1.plot(
        df_resultados["k"],
        df_resultados["dunn_serial"],
        "o-",
        label="Serial",
        color="blue",
    )
    ax1.plot(
        df_resultados["k"],
        df_resultados["dunn_parallel"],
        "s--",
        label="Paralelo",
        color="red",
    )
    ax1.set_xlabel("Número de Clusters (k)")
    ax1.set_ylabel("Dunn Index")
    ax1.set_title("Dunn Index vs Número de Clusters")
    ax1.legend()
    ax1.grid(True)

    # Tempo de execução vs k
    ax2.plot(
        df_resultados["k"],
        df_resultados["tempo_serial"],
        "o-",
        label="Serial",
        color="skyblue",
    )
    ax2.plot(
        df_resultados["k"],
        df_resultados["tempo_parallel"],
        "s--",
        label="Paralelo",
        color="orange",
    )
    ax2.set_xlabel("Número de Clusters (k)")
    ax2.set_ylabel("Tempo (s)")
    ax2.set_title("Tempo de Execução vs Número de Clusters")
    ax2.legend()
    ax2.grid(True)

    # Speedup vs k
    ax3.plot(df_resultados["k"], df_resultados["speedup"], "o-", color="green")
    ax3.set_xlabel("Número de Clusters (k)")
    ax3.set_ylabel("Speedup")
    ax3.set_title("Speedup (Serial/Paralelo) vs Número de Clusters")
    ax3.grid(True)

    # Comparação de tempos
    x = np.arange(len(df_resultados["k"]))
    width = 0.35
    ax4.bar(
        x - width / 2,
        df_resultados["tempo_serial"],
        width,
        label="Serial",
        color="skyblue",
    )
    ax4.bar(
        x + width / 2,
        df_resultados["tempo_parallel"],
        width,
        label="Paralelo",
        color="orange",
    )
    ax4.set_xlabel("Número de Clusters (k)")
    ax4.set_ylabel("Tempo (s)")
    ax4.set_title("Comparação de Tempos de Execução")
    ax4.set_xticks(x)
    ax4.set_xticklabels(df_resultados["k"])
    ax4.legend()
    ax4.grid(axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "analise_comparativa.png"), dpi=300)
    plt.show()

    # Imprimir resumo final
    print(f"\n{'=' * 60}")
    print("📊 RESUMO FINAL DOS EXPERIMENTOS")
    print(f"{'=' * 60}")
    print(df_resultados.to_string(index=False, float_format="%.4f"))
    print(f"\n✅ Todos os resultados salvos em: {results_dir}/")
    print(
        f"📈 Melhor Dunn Index: {df_resultados['dunn_serial'].max():.4f} (k = {df_resultados.loc[df_resultados['dunn_serial'].idxmax(), 'k']})"
    )
    print(
        f"⚡ Melhor Speedup: {df_resultados['speedup'].max():.2f}x (k = {df_resultados.loc[df_resultados['speedup'].idxmax(), 'k']})"
    )
