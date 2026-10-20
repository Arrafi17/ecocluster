import pandas as pd
from sklearn.metrics import davies_bouldin_score, silhouette_score


def evaluasi(x, label):
    return {
        "silhouette": round(float(silhouette_score(x, label)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(x, label)), 4),
    }


def bandingkan_latih_uji(model, x_latih, x_uji):
    e_latih = evaluasi(x_latih, model.labels_)
    e_uji = evaluasi(x_uji, model.predict(x_uji))
    hasil = pd.DataFrame({
        "data": ["latih", "uji"],
        "jumlah": [len(x_latih), len(x_uji)],
        "silhouette": [e_latih["silhouette"], e_uji["silhouette"]],
        "davies_bouldin": [e_latih["davies_bouldin"], e_uji["davies_bouldin"]],
    })
    selisih = abs(e_latih["silhouette"] - e_uji["silhouette"])
    status = "stabil" if selisih < 0.1 else "perlu diperiksa (indikasi overfitting)"
    return hasil, round(float(selisih), 4), status


def profil_cluster(df, label, fitur):
    salinan = df.copy()
    salinan["cluster"] = label
    profil = salinan.groupby("cluster")[fitur].mean().round(2)
    profil["jumlah_wilayah"] = salinan.groupby("cluster").size()
    return profil


def beri_nama(profil):
    nama = {}
    sisa = list(profil.index)

    padat = profil.loc[sisa, "kepadatan_jiwa_km2"].idxmax()
    nama[padat] = "Padat dan tertekan"
    sisa.remove(padat)

    if sisa:
        hijau = profil.loc[sisa, "tutupan_hijau_pct"].idxmax()
        nama[hijau] = "Hijau dan sejuk"
        sisa.remove(hijau)

    for urut, c in enumerate(sisa, start=1):
        nama[c] = "Transisi pesisir-agraris" if len(sisa) == 1 else f"Transisi {urut}"
    return nama
