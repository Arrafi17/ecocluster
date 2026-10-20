import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score

from .config import K_RANGE, RANDOM_STATE


def cari_k(x):
    baris = []
    for k in K_RANGE:
        model = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x)
        baris.append({
            "k": k,
            "inertia": model.inertia_,
            "silhouette": silhouette_score(x, model.labels_),
            "davies_bouldin": davies_bouldin_score(x, model.labels_),
        })
    tabel = pd.DataFrame(baris)
    k_terbaik = int(tabel.loc[tabel["silhouette"].idxmax(), "k"])
    return tabel, k_terbaik


def latih_model(x, k):
    return KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(x)
