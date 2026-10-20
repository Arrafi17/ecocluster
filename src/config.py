from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "indikator_lingkungan.csv"
OUTPUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
N_WILAYAH = 72
TEST_SIZE = 0.25
K_RANGE = range(2, 9)

FITUR = [
    "indeks_udara",
    "tutupan_hijau_pct",
    "curah_hujan_mm",
    "suhu_rata_c",
    "kepadatan_jiwa_km2",
    "sampah_kg_per_jiwa",
    "indeks_air",
]
