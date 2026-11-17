"""Tahap 1 Metodologi: Web scraping 6 portal berita Arab, kompilasi korpus, dan statistik dinamis."""

import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import unicodedata
from urllib.parse import urljoin

from bs4 import BeautifulSoup
import pandas as pd

from .config import (
    LABELS,
    LABELS_ID,
    LABEL_TO_ID,
    PORTAL_METADATA,
    STATS_DIR,
    resolve_data_file,
)

# Konfigurasi pola URL dan selektor CSS HTML untuk ke-6 portal berita Arab
PORTAL_SCRAPER_CONFIG: Dict[str, Dict[str, Any]] = {
    "alsumaria": {
        "domain": PORTAL_METADATA["alsumaria"]["source"],
        "country": PORTAL_METADATA["alsumaria"]["country"],
        "url_template": (
            "https://www.alsumaria.tv/Website/DynamicPages/LoadMore/"
            "Loadmore_DocumentCategoryTemp2.aspx?loadindex={page_no}&lang=ar&ID={category_id}"
        ),
        "link_selector": "a[href]",
        "href_filter": None,
        "link_prefix": "https://www.alsumaria.tv",
        "title_selector": "h1 .ReadMyTitle",
        "content_selector": ".ReadThisText",
        "drop_last_paragraph": False,
        "paragraph_join": "\n",
    },
    "alsharq": {
        "domain": PORTAL_METADATA["alsharq"]["source"],
        "country": PORTAL_METADATA["alsharq"]["country"],
        "url_template": "https://al-sharq.com/category/{category_id}/{page_no}",
        "link_selector": ".ec-arbic a.title",
        "href_filter": None,
        "link_prefix": "https://al-sharq.com",
        "title_selector": ".minit--1 h1.title",
        "content_selector": ".minit--1 > p",
        "drop_last_paragraph": False,
        "paragraph_join": "\n",
    },
    "assahraa": {
        "domain": PORTAL_METADATA["assahraa"]["source"],
        "country": PORTAL_METADATA["assahraa"]["country"],
        "url_template": "https://assahraa.ma/sujet/{category_id}?page={page_no}",
        "link_selector": "#_items div[data-key] a",
        "href_filter": None,
        "link_prefix": "https://assahraa.ma/",
        "title_selector": ".sec-info h1",
        "content_selector": ".article p",
        "drop_last_paragraph": True,
        "paragraph_join": " ",
    },
    "akhbarelsyaha": {
        "domain": PORTAL_METADATA["akhbarelsyaha"]["source"],
        "country": PORTAL_METADATA["akhbarelsyaha"]["country"],
        "url_template": "https://akhbarelsyaha.com/magazine/{category_id}/page/{page_no}",
        "link_selector": "h2 a",
        "href_filter": None,
        "link_prefix": "",
        "title_selector": "h1.post-title",
        "content_selector": ".entry-content p",
        "drop_last_paragraph": False,
        "paragraph_join": "\n",
    },
    "saraya": {
        "domain": PORTAL_METADATA["saraya"]["source"],
        "country": PORTAL_METADATA["saraya"]["country"],
        "url_template": "https://www.sarayanews.com/category/{category_id}/{page_no}",
        "link_selector": ".newslstsep h1 a",
        "href_filter": None,
        "link_prefix": "",
        "title_selector": "h2.title",
        "content_selector": "#newscontent p:first-child",
        "drop_last_paragraph": False,
        "paragraph_join": "\n",
    },
    "addustour": {
        "domain": PORTAL_METADATA["addustour"]["source"],
        "country": PORTAL_METADATA["addustour"]["country"],
        "url_template": "https://www.addustour.com/categories/{category_id}/page{page_no}",
        "link_selector": ".page-content li.list-item a",
        "href_filter": "articles",
        "link_prefix": "",
        "title_selector": "h1.headingInfo_title",
        "content_selector": "#body-text p",
        "drop_last_paragraph": False,
        "paragraph_join": "\n",
    },
}


def parse_category_urls_html(html_content: str, portal: str) -> List[str]:
    """Mengekstrak daftar URL artikel dari HTML halaman indeks kategori portal berita."""
    if portal not in PORTAL_SCRAPER_CONFIG:
        raise ValueError(f"Portal tidak dikenal: {portal}")
    cfg = PORTAL_SCRAPER_CONFIG[portal]
    soup = BeautifulSoup(html_content, "html.parser")
    links = soup.select(cfg["link_selector"])

    urls: List[str] = []
    href_filter = cfg["href_filter"]
    prefix = cfg["link_prefix"]
    for link in links:
        href = link.get("href")
        if not href:
            continue
        if href_filter and href_filter not in href:
            continue
        full_url = urljoin(prefix, href) if prefix and not href.startswith("http") else href
        urls.append(full_url)
    return urls


def fetch_category_article_urls(
    portal: str,
    category_slug: Union[str, int],
    page_no: int,
    session: Any = None,
    timeout: int = 15,
) -> List[str]:
    """Mengambil URL artikel dari satu halaman indeks kategori portal berita."""
    if portal not in PORTAL_SCRAPER_CONFIG:
        raise ValueError(f"Portal tidak dikenal: {portal}")
    import requests

    cfg = PORTAL_SCRAPER_CONFIG[portal]
    url = cfg["url_template"].format(category_id=category_slug, page_no=page_no)
    client = session if session is not None else requests
    response = client.get(url, timeout=timeout)
    if response.status_code == 204 or response.status_code != 200:
        return []
    return parse_category_urls_html(response.text, portal=portal)


def scrape_article_html(html_content: Union[str, bytes], portal: str) -> Tuple[str, str]:
    """Mengekstrak judul dan isi artikel dari HTML halaman berita."""
    if portal not in PORTAL_SCRAPER_CONFIG:
        raise ValueError(f"Portal tidak dikenal: {portal}")
    cfg = PORTAL_SCRAPER_CONFIG[portal]
    soup = BeautifulSoup(html_content, "html.parser")

    title_el = soup.select_one(cfg["title_selector"])
    title = title_el.get_text(strip=True) if title_el else ""

    p_elements = soup.select(cfg["content_selector"])
    if cfg["drop_last_paragraph"] and p_elements:
        p_elements = p_elements[:-1]
    body = cfg["paragraph_join"].join(el.get_text(strip=True) for el in p_elements)
    return title, body


def scrape_and_save_article(
    url: str,
    portal: str,
    output_dir: Union[str, Path],
    index: int,
    session: Any = None,
    timeout: int = 15,
) -> Path:
    """Mengunduh satu artikel dan menyimpannya ke berkas teks `<index>.txt`."""
    import requests

    client = session if session is not None else requests
    response = client.get(url, timeout=timeout)
    response.raise_for_status()
    title, body = scrape_article_html(response.content, portal=portal)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{index}.txt"
    with out_file.open("w", encoding="utf-8") as handle:
        if title:
            handle.write(f"{title}\n\n")
        if body:
            handle.write(f"{body}\n")
    return out_file


def read_articles_from_folder(
    base_dir: Union[str, Path],
    category_name: str,
    source_name: str,
) -> List[Dict[str, Any]]:
    """Membaca berkas teks hasil scraping pada struktur direktori `<kategori>/<sumber>/`."""
    folder_path = Path(base_dir) / category_name / source_name
    if not folder_path.exists():
        return []

    label_id = LABEL_TO_ID[unicodedata.normalize("NFC", category_name)]
    articles: List[Dict[str, Any]] = []
    for file_path in sorted(folder_path.iterdir()):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        content = file_path.read_text(encoding="utf-8").replace("\n", "").replace("\t", "")
        articles.append(
            {
                "label": label_id,
                "category": category_name,
                "filename": file_path.name,
                "source": source_name,
                "text": content,
            }
        )
    return articles


def compile_raw_articles_dataset(
    base_dir: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    sep: str = ",",
) -> pd.DataFrame:
    """Menggabungkan seluruh berkas teks menjadi satu DataFrame korpus terstruktur."""
    dataset: List[Dict[str, Any]] = []
    for category_name in LABELS.values():
        for source_name in PORTAL_METADATA.keys():
            dataset.extend(read_articles_from_folder(base_dir, category_name, source_name))
    df = pd.DataFrame(dataset, columns=["label", "category", "filename", "source", "text"])
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False, sep=sep, encoding="utf-8")
    return df


def compute_corpus_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Menghitung distribusi artikel per kategori dan per portal secara dinamis dari DataFrame."""
    total_raw = int(len(df))
    valid_df = df.dropna(subset=["text"]).copy()
    valid_df = valid_df[valid_df["text"].astype(str).str.strip() != ""]
    total_valid = int(len(valid_df))
    empty_rows = total_raw - total_valid

    raw_cat_counts = df["category"].astype(str).apply(lambda s: unicodedata.normalize("NFC", s)).value_counts()
    valid_cat_counts = (
        valid_df["category"].astype(str).apply(lambda s: unicodedata.normalize("NFC", s)).value_counts()
    )

    cat_rows: List[Dict[str, Any]] = []
    for cid, cname in LABELS.items():
        norm_c = unicodedata.normalize("NFC", cname)
        v_cnt = int(valid_cat_counts.get(norm_c, 0))
        r_cnt = int(raw_cat_counts.get(norm_c, 0))
        cat_rows.append(
            {
                "Label": cid,
                "Kategori": cname,
                "Terjemahan": LABELS_ID[cid],
                "Artikel Valid": v_cnt,
                "Persentase (%)": round(v_cnt / total_valid * 100, 1) if total_valid else 0.0,
                "Artikel Mentah": r_cnt,
                "Baris Kosong": r_cnt - v_cnt,
            }
        )
    cat_df = pd.DataFrame(cat_rows)

    src_counts = valid_df["source"].astype(str).value_counts()
    src_rows: List[Dict[str, Any]] = []
    for key, meta in PORTAL_METADATA.items():
        cnt = int(src_counts.get(key, 0))
        src_rows.append(
            {
                "Kode Sumber": key,
                "Portal Berita": meta["source"],
                "Asal Negara": meta["country"],
                "Jumlah Artikel": cnt,
                "Persentase (%)": round(cnt / total_valid * 100, 1) if total_valid else 0.0,
            }
        )
    src_df = pd.DataFrame(src_rows).sort_values("Jumlah Artikel", ascending=False).reset_index(drop=True)

    return {
        "total_raw_rows": total_raw,
        "empty_text_rows": empty_rows,
        "total_valid_articles": total_valid,
        "category_distribution": cat_df,
        "source_distribution": src_df,
    }


def compute_token_counts_per_category(
    df: pd.DataFrame,
    text_col: str = "text",
    category_col: str = "category",
) -> Dict[str, int]:
    """Menghitung total token per kategori secara dinamis dari kolom teks korpus."""
    work = df[[category_col, text_col]].copy()
    work[text_col] = work[text_col].fillna("").astype(str)
    counts = work.groupby(category_col)[text_col].apply(lambda col: len(" ".join(col).split())).to_dict()
    return {unicodedata.normalize("NFC", str(k)): int(v) for k, v in counts.items()}


def compute_word_type_counts(df: pd.DataFrame, text_col: str = "text") -> Dict[str, int]:
    """Menghitung jumlah tipe kata unik Arab vs Non-Arab menggunakan pencocokan ekspresi reguler."""
    joined = " ".join(df[text_col].fillna("").astype(str))
    arabic_types = len(set(re.findall(r"\b[\u0600-\u06FF]+\b", joined)))
    nonarabic_types = len(set(re.findall(r"\b(?![\u0600-\u06FF])\w+\b", joined)))
    total_tokens = len(joined.split())
    return {
        "total_tokens": int(total_tokens),
        "arabic_word_types": int(arabic_types),
        "nonarabic_word_types": int(nonarabic_types),
        "repeated_tokens_diff": int(total_tokens - arabic_types - nonarabic_types),
    }


def find_file_unicode_safe(
    directory: Union[str, Path],
    prefix: str,
    suffix: str = ".csv",
) -> Path:
    """Mencari berkas bernama huruf Arab secara aman terhadap perbedaan normalisasi Unicode NFC vs NFD."""
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"Direktori belum dibuat: {base}")
    target_prefix = unicodedata.normalize("NFC", prefix)
    for path in sorted(base.iterdir()):
        norm_name = unicodedata.normalize("NFC", path.name)
        if norm_name.startswith(target_prefix) and norm_name.endswith(suffix):
            return path
    raise FileNotFoundError(f"File '{prefix}*{suffix}' tidak ditemukan di {base}")


def save_corpus_summary_json(
    stats_payload: Dict[str, Any],
    stats_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Menyimpan hasil perhitungan statistik korpus ke results/stats/corpus_summary.json."""
    s_dir = Path(stats_dir) if stats_dir else STATS_DIR
    s_dir.mkdir(parents=True, exist_ok=True)
    serializable = dict(stats_payload)
    if isinstance(serializable.get("category_distribution"), pd.DataFrame):
        serializable["category_distribution"] = serializable["category_distribution"].to_dict(orient="records")
    if isinstance(serializable.get("source_distribution"), pd.DataFrame):
        serializable["source_distribution"] = serializable["source_distribution"].to_dict(orient="records")
    serializable.pop("source_mode", None)

    out_path = s_dir / "corpus_summary.json"
    out_path.write_text(json.dumps(serializable, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def load_or_compute_corpus_stats(
    dataset_path: Optional[Union[str, Path]] = None,
    stats_dir: Optional[Union[str, Path]] = None,
    prefer_summary: bool = False,
) -> Dict[str, Any]:
    """Menghitung statistik korpus langsung dari basic_articles_dataset.csv atau memuat ringkasan hasil notebook."""
    s_dir = Path(stats_dir) if stats_dir else STATS_DIR
    summary_json = s_dir / "corpus_summary.json"

    if prefer_summary and summary_json.exists():
        data = json.loads(summary_json.read_text(encoding="utf-8"))
        return {
            "source_mode": "loaded_from_results_summary",
            "total_raw_rows": data.get("total_raw_rows", 0),
            "empty_text_rows": data.get("empty_text_rows", 0),
            "total_valid_articles": data.get("total_valid_articles", 0),
            "category_distribution": pd.DataFrame(data.get("category_distribution", [])),
            "source_distribution": pd.DataFrame(data.get("source_distribution", [])),
            "tokens_per_category": data.get("tokens_per_category", {}),
            "nostop_tokens_per_category": data.get("nostop_tokens_per_category", {}),
            "total_tokens": data.get("total_tokens", 0),
            "arabic_word_types": data.get("arabic_word_types", 0),
            "nonarabic_word_types": data.get("nonarabic_word_types", 0),
            "repeated_tokens_diff": data.get("repeated_tokens_diff", 0),
        }

    basic_csv = Path(dataset_path) if dataset_path else resolve_data_file("basic_articles_dataset.csv")
    if basic_csv.exists():
        df = pd.read_csv(basic_csv, low_memory=False)
        dist = compute_corpus_distributions(df)
        tokens_per_cat = compute_token_counts_per_category(df)
        word_types = compute_word_type_counts(df)
        return {
            "source_mode": "computed_from_csv",
            **dist,
            "tokens_per_category": tokens_per_cat,
            **word_types,
        }

    if summary_json.exists():
        return load_or_compute_corpus_stats(dataset_path=dataset_path, stats_dir=s_dir, prefer_summary=True)

    return {"source_mode": "not_generated_yet"}
