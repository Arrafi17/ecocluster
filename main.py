import pandas as pd

from src import clustering, evaluation, preprocessing, visualization
from src.config import DATA_PATH, OUTPUT_DIR
from src.data_generator import simpan_dataset


def garis(judul):
    print("\n" + "=" * 60)
    print(judul)
    print("=" * 60)


def main():
    pd.set_option("display.width", 140)
    OUTPUT_DIR.mkdir(exist_ok=True)

    garis("1. PENGUMPULAN DATA")
    if not DATA_PATH.exists():
        simpan_dataset()
    df = preprocessing.muat_data()
    print(f"{df.shape[0]} wilayah, {df.shape[1] - 1} indikator")
    print(preprocessing.ringkasan_kualitas(df))

    garis("2. PRA-PEMROSESAN")
    df = preprocessing.bersihkan(df)
    print("Nilai kosong setelah imputasi median:", int(df.isna().sum().sum()))

    garis("3. PEMILIHAN FITUR")
    fitur, korelasi = preprocessing.pilih_fitur(df)
    print("Fitur dipakai:", fitur)

    garis("4. PEMBAGIAN DATA")
    latih, uji = preprocessing.bagi_data(df)
    scaler, x_latih, x_uji = preprocessing.skala(latih, uji, fitur)
    print(f"Data latih: {len(latih)} | Data uji: {len(uji)}")

    garis("5. PELATIHAN MODEL (k-Means)")
    tabel_k, k = clustering.cari_k(x_latih)
    print(tabel_k.round(4).to_string(index=False))
    print(f"\nk terpilih (silhouette tertinggi): {k}")
    model = clustering.latih_model(x_latih, k)
    visualization.plot_pemilihan_k(tabel_k, k)

    garis("6. EVALUASI MODEL")
    hasil, selisih, status = evaluation.bandingkan_latih_uji(model, x_latih, x_uji)
    print(hasil.to_string(index=False))
    print(f"Selisih silhouette latih-uji: {selisih} -> {status}")

    profil = evaluation.profil_cluster(latih, model.labels_, fitur)
    nama = evaluation.beri_nama(profil)
    print("\nProfil cluster:")
    print(profil.rename(index=nama).to_string())

    visualization.plot_pca(x_latih, model.labels_, nama, model.cluster_centers_)
    visualization.plot_heatmap_profil(profil, fitur, nama)
    visualization.plot_ukuran_cluster(profil, nama)

    garis("7. PREDIKSI WILAYAH BARU")
    baru = pd.DataFrame([
        {"wilayah": "Contoh-A", "indeks_udara": 90, "tutupan_hijau_pct": 70, "curah_hujan_mm": 3100,
         "suhu_rata_c": 24, "kepadatan_jiwa_km2": 200, "sampah_kg_per_jiwa": 0.3, "indeks_air": 85},
        {"wilayah": "Contoh-B", "indeks_udara": 50, "tutupan_hijau_pct": 10, "curah_hujan_mm": 2000,
         "suhu_rata_c": 31, "kepadatan_jiwa_km2": 5500, "sampah_kg_per_jiwa": 1.0, "indeks_air": 45},
    ])
    prediksi = model.predict(scaler.transform(baru[fitur]))
    baru["cluster"] = prediksi
    baru["karakter"] = [nama[c] for c in prediksi]
    print(baru[["wilayah", "cluster", "karakter"]].to_string(index=False))

    penuh = scaler.transform(df[fitur])
    df_hasil = df.copy()
    df_hasil["cluster"] = model.predict(penuh)
    df_hasil["karakter"] = df_hasil["cluster"].map(nama)
    df_hasil.to_csv(OUTPUT_DIR / "hasil_clustering.csv", index=False)
    profil.rename(index=nama).to_csv(OUTPUT_DIR / "profil_cluster.csv")
    tabel_k.to_csv(OUTPUT_DIR / "evaluasi_k.csv", index=False)
    print(f"\nSemua keluaran tersimpan di: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
