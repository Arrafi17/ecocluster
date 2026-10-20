import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

from .config import OUTPUT_DIR

WARNA = ["#2a9d8f", "#e76f51", "#264653", "#e9c46a", "#8d6cab", "#4d908e", "#f4a261", "#577590"]


def _simpan(nama):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / nama, dpi=140)
    plt.close()


def plot_pemilihan_k(tabel, k_terbaik):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(tabel["k"], tabel["inertia"], marker="o", color=WARNA[0])
    ax[0].set_title("Elbow (Inertia)")
    ax[1].plot(tabel["k"], tabel["silhouette"], marker="o", color=WARNA[1])
    ax[1].axvline(k_terbaik, ls="--", color="gray")
    ax[1].set_title("Silhouette Score")
    ax[2].plot(tabel["k"], tabel["davies_bouldin"], marker="o", color=WARNA[2])
    ax[2].set_title("Davies-Bouldin Index")
    for a in ax:
        a.set_xlabel("Jumlah cluster (k)")
        a.grid(alpha=0.3)
    _simpan("01_pemilihan_k.png")


def plot_pca(x, label, nama, pusat=None):
    pca = PCA(n_components=2, random_state=42).fit(x)
    z = pca.transform(x)
    plt.figure(figsize=(7, 5.5))
    for c in np.unique(label):
        m = label == c
        plt.scatter(z[m, 0], z[m, 1], s=55, color=WARNA[c % len(WARNA)],
                    label=nama.get(c, f"Cluster {c}"), edgecolor="white")
    if pusat is not None:
        zp = pca.transform(pusat)
        plt.scatter(zp[:, 0], zp[:, 1], marker="X", s=220, color="black", label="Centroid")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)")
    plt.title("Sebaran Wilayah per Cluster (proyeksi PCA)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    _simpan("02_sebaran_pca.png")


def plot_heatmap_profil(profil, fitur, nama):
    data = profil[fitur]
    z = (data - data.mean()) / data.std(ddof=0)
    plt.figure(figsize=(10, 4))
    plt.imshow(z.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=2)
    plt.xticks(range(len(fitur)), fitur, rotation=30, ha="right")
    plt.yticks(range(len(z)), [nama[c] for c in z.index])
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            plt.text(j, i, f"{data.values[i, j]:g}", ha="center", va="center", fontsize=8)
    plt.title("Profil Rata-rata Fitur per Cluster")
    plt.colorbar(label="skor relatif")
    _simpan("03_profil_cluster.png")


def plot_ukuran_cluster(profil, nama):
    plt.figure(figsize=(7, 4))
    label = [nama[c] for c in profil.index]
    plt.bar(label, profil["jumlah_wilayah"], color=WARNA[:len(label)])
    plt.ylabel("Jumlah wilayah")
    plt.xticks(rotation=15)
    plt.title("Ukuran Tiap Cluster")
    _simpan("04_ukuran_cluster.png")
