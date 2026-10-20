import numpy as np
import pandas as pd

from .config import DATA_PATH, N_WILAYAH, RANDOM_STATE

PROFIL = {
    "pesisir": dict(
        indeks_udara=(78, 6), tutupan_hijau_pct=(28, 7), curah_hujan_mm=(2300, 250),
        suhu_rata_c=(29.5, 0.6), kepadatan_jiwa_km2=(900, 250), sampah_kg_per_jiwa=(0.55, 0.1),
        indeks_air=(62, 7),
    ),
    "perkotaan": dict(
        indeks_udara=(52, 7), tutupan_hijau_pct=(12, 4), curah_hujan_mm=(2000, 220),
        suhu_rata_c=(30.8, 0.6), kepadatan_jiwa_km2=(5200, 900), sampah_kg_per_jiwa=(0.95, 0.12),
        indeks_air=(48, 7),
    ),
    "pegunungan": dict(
        indeks_udara=(92, 4), tutupan_hijau_pct=(74, 8), curah_hujan_mm=(3200, 300),
        suhu_rata_c=(23.5, 1.2), kepadatan_jiwa_km2=(180, 70), sampah_kg_per_jiwa=(0.25, 0.07),
        indeks_air=(88, 5),
    ),
    "pertanian": dict(
        indeks_udara=(82, 5), tutupan_hijau_pct=(48, 8), curah_hujan_mm=(2500, 260),
        suhu_rata_c=(27.2, 0.8), kepadatan_jiwa_km2=(450, 130), sampah_kg_per_jiwa=(0.4, 0.08),
        indeks_air=(70, 7),
    ),
}


def buat_dataset(n=N_WILAYAH, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    tipe = list(PROFIL)
    bobot = [0.22, 0.25, 0.23, 0.30]
    baris = []
    for i in range(n):
        asal = rng.choice(tipe, p=bobot)
        nilai = {k: rng.normal(m, s) for k, (m, s) in PROFIL[asal].items()}
        nilai["indeks_udara"] = float(np.clip(nilai["indeks_udara"], 20, 100))
        nilai["tutupan_hijau_pct"] = float(np.clip(nilai["tutupan_hijau_pct"], 2, 95))
        nilai["indeks_air"] = float(np.clip(nilai["indeks_air"], 20, 100))
        nilai["kepadatan_jiwa_km2"] = float(max(nilai["kepadatan_jiwa_km2"], 50))
        nilai["sampah_kg_per_jiwa"] = float(max(nilai["sampah_kg_per_jiwa"], 0.1))
        baris.append({"wilayah": f"Wilayah-{i + 1:02d}", **nilai})
    df = pd.DataFrame(baris).round(2)
    mask = rng.random(df.shape[0]) < 0.04
    df.loc[mask, "indeks_air"] = np.nan
    return df


def simpan_dataset():
    df = buat_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df
