import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import DATA_PATH, FITUR, RANDOM_STATE, TEST_SIZE


def muat_data():
    return pd.read_csv(DATA_PATH)


def ringkasan_kualitas(df):
    return pd.DataFrame({
        "tipe": df.dtypes.astype(str),
        "kosong": df.isna().sum(),
        "unik": df.nunique(),
    })


def bersihkan(df):
    hasil = df.drop_duplicates(subset="wilayah").copy()
    for kolom in FITUR:
        hasil[kolom] = hasil[kolom].fillna(hasil[kolom].median())
    return hasil


def pilih_fitur(df, ambang_korelasi=0.9):
    korelasi = df[FITUR].corr().abs()
    dibuang = set()
    for i, a in enumerate(FITUR):
        for b in FITUR[i + 1:]:
            if korelasi.loc[a, b] > ambang_korelasi and b not in dibuang:
                dibuang.add(b)
    return [f for f in FITUR if f not in dibuang], korelasi


def bagi_data(df):
    latih, uji = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    return latih.reset_index(drop=True), uji.reset_index(drop=True)


def skala(latih, uji, fitur):
    scaler = StandardScaler().fit(latih[fitur])
    return scaler, scaler.transform(latih[fitur]), scaler.transform(uji[fitur])
