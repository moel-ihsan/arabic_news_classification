"""Konfigurasi path relatif, resolusi file data tunggal, konstanta label, dan random seed."""

from pathlib import Path

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_DIR: Path = REPO_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = REPO_ROOT / "models"
FRAME_MODELS_DIR: Path = MODELS_DIR / "frame_semantiks"

RESULTS_DIR: Path = REPO_ROOT / "results"
METRICS_DIR: Path = RESULTS_DIR / "metrics"
STATS_DIR: Path = RESULTS_DIR / "stats"
LEXICONS_DIR: Path = RESULTS_DIR / "lexicons"
FIGURES_DIR: Path = RESULTS_DIR / "figures"
DOCS_DIR: Path = REPO_ROOT / "docs"

RANDOM_SEED: int = 42


def resolve_data_file(filename: str) -> Path:
    """Mencari berkas data di data/ terlebih dahulu, lalu fallback ke data/processed/ atau data/raw/."""
    for base in (DATA_DIR, PROCESSED_DATA_DIR, RAW_DATA_DIR):
        candidate = base / filename
        if candidate.exists():
            return candidate
    return DATA_DIR / filename


LABELS = {
    1: "أمن",
    2: "اقتصاد",
    3: "تكنولوجيا",
    4: "ثقافة",
    5: "دوليات",
    6: "رياضة",
    7: "سياحة",
    8: "سياسة",
    9: "صحة",
    10: "علم",
    11: "مجتمع",
}

LABEL_TO_ID = {name: label_id for label_id, name in LABELS.items()}

LABELS_ID = {
    1: "Keamanan",
    2: "Ekonomi",
    3: "Teknologi",
    4: "Budaya",
    5: "Internasional",
    6: "Olahraga",
    7: "Travel / Pariwisata",
    8: "Politik",
    9: "Kesehatan",
    10: "Sains",
    11: "Masyarakat / Komunitas",
}

LABELS_EN = {
    1: "Security",
    2: "Economy",
    3: "Technology",
    4: "Culture",
    5: "International",
    6: "Sports",
    7: "Travel",
    8: "Politics",
    9: "Health",
    10: "Science",
    11: "Society",
}

# Hanya metadata struktural sumber berita (jumlah & persentase dihitung dinamis dari dataset)
PORTAL_METADATA = {
    "alsumaria": {"source": "alsumaria.tv", "country": "Irak"},
    "alsharq": {"source": "al-sharq.com", "country": "Qatar"},
    "assahraa": {"source": "assahraa.ma", "country": "Maroko"},
    "akhbarelsyaha": {"source": "akhbarelsyaha.com", "country": "Mesir"},
    "saraya": {"source": "sarayanews.com", "country": "Yordania"},
    "addustour": {"source": "addustour.com", "country": "Yordania"},
}

SOURCES_INFO = [
    {"key": key, **meta} for key, meta in PORTAL_METADATA.items()
]
