# Klasifikasi Artikel Berita Berbahasa Arab Berbasis Medan Makna (*Semantic Field*) dan Semantik *Frame* (*Frame Semantics*)

Repositori resmi penelitian komputasi linguistik dan *Natural Language Processing* (NLP) untuk klasifikasi multi-kelas (**11 kategori**) pada korpus **170.934 artikel berita daring berbahasa Arab** dari **6 portal berita Timur Tengah dan Afrika Utara**. Proyek ini membandingkan secara sistematis dua pendekatan representasi semantik — **Medan Makna (*Unigram*, `ngram_range=(1, 1)`)** dan **Semantik *Frame* (*Bigram & Trigram*, `ngram_range=(2, 3)`)** — serta menyediakan antarmuka demo interaktif berbasis **Streamlit** dengan dukungan penuh teks *Right-to-Left* (RTL).

---

## Latar Belakang Penelitian

Pemrosesan teks bahasa Arab menghadapi tantangan morfologis dan polisemi yang tinggi ketika kata diperlakukan sebagai unit tunggal yang terlepas dari konteks kalimatnya. Penelitian ini mengevaluasi perbedaan efektivitas dan efisiensi komputasi antara dua representasi fitur TF-IDF pada lima algoritma pembelajaran mesin (*Support Vector Machine*, *Logistic Regression*, *Naïve Bayes*, *Random Forest*, dan *Gradient Boosting*):

1. **Pendekatan Medan Makna (*Semantic Field* — `ngram_range=(1, 1)`)**:
   - Menerapkan penghapusan kata henti (*stopword removal*) untuk menyaring kata leksikal bermakna inti yang merepresentasikan domain topik artikel (`80%` latih / `15%` validasi / `5%` uji).
   - Model terbaik **SVM** mencapai akurasi **`90,26%`** pada korpus penuh (`N = 25.641`) dan **`90,47%`** pada eksperimen publikasi jurnal awal (`N = 152.793` artikel).
2. **Pendekatan Semantik *Frame* (*Frame Semantics* — `ngram_range=(2, 3)`)**:
   - **Mempertahankan kata tugas dan preposisi Arab** (`في`, `من`, `على`, `إلى`, `أن`) karena kata penghubung tersebut membentuk relasi kontekstual 2–3 kata (*semantic frame*) yang membedakan konteks berita secara lebih akurat (`85%` latih / `10%` validasi / `5%` uji).
   - Model terbaik **SVM (`LinearSVC C=10`)** meningkatkan akurasi menjadi **`90,61%`** (`Precision = 0,9088`, `F1-Score = 0,9068`, `N = 17.096`), dengan akurasi kategori tertinggi pada Olahraga (`F1 = 0,9879`).

---

## Karakteristik Korpus Berita Arab

Seluruh data dikumpulkan melalui *web scraping* bertahap (`requests` & `BeautifulSoup`) dari enam portal berita Arab dan dikompilasi ke dalam dataset tunggal `basic_articles_dataset.csv` (`171.175` berkas mentah; **`170.934` artikel valid non-null**; **`42.661.503` token mentah**; **`30.559.650` token pasca-*stopword removal***; **`754.584` tipe kata unik Arab**; dan **`74.727` tipe kata Non-Arab**).

### Distribusi 6 Portal Berita

| Kode | Portal Berita | Negara | Jumlah Artikel | Persentase |
|---|---|---|---:|---:|
| `alsumaria` | `alsumaria.tv` | Irak | `73.196` | `42,8%` |
| `alsharq` | `al-sharq.com` | Qatar | `41.159` | `24,1%` |
| `assahraa` | `assahraa.ma` | Maroko | `28.601` | `16,7%` |
| `akhbarelsyaha` | `akhbarelsyaha.com` | Mesir | `15.002` | `8,8%` |
| `saraya` | `sarayanews.com` | Yordania | `7.805` | `4,6%` |
| `addustour` | `addustour.com` | Yordania | `5.171` | `3,0%` |
| **Total** | — | — | **`170.934`** | **`100,0%`** |

### Distribusi 11 Kategori

| Label | Kategori (Arab) | Terjemahan | Artikel Valid | Persentase | Total Token |
|---:|---|---|---:|---:|---:|
| `1` | `أمن` | Keamanan | `17.341` | `10,1%` | `4.102.495` |
| `2` | `اقتصاد` | Ekonomi | `15.899` | `9,3%` | `3.636.123` |
| `3` | `تكنولوجيا` | Teknologi | `11.036` | `6,5%` | `2.751.310` |
| `4` | `ثقافة` | Budaya | `10.844` | `6,3%` | `2.086.216` |
| `5` | `دوليات` | Internasional | `17.964` | `10,5%` | `5.608.498` |
| `6` | `رياضة` | Olahraga | `18.612` | `10,9%` | `5.476.537` |
| `7` | `سياحة` | Travel / Pariwisata | `18.250` | `10,7%` | `5.608.691` |
| `8` | `سياسة` | Politik | `18.295` | `10,7%` | `3.895.209` |
| `9` | `صحة` | Kesehatan | `13.260` | `7,8%` | `3.004.546` |
| `10` | `علم` | Sains | `14.199` | `8,3%` | `3.438.063` |
| `11` | `مجتمع` | Masyarakat / Komunitas | `15.234` | `8,9%` | `3.053.815` |

---

## Struktur Repositori

```text
arabic_news_classification/
├── app/
│   └── streamlit_app.py                      # Aplikasi demo interaktif Streamlit (dukungan RTL Arab)
├── data/
│   ├── stopwords.txt                         # Daftar kata henti Arab (digunakan jalur Medan Makna)
│   └── basic_articles_dataset.csv            # Dataset korpus utama (lokal, dikecualikan dari Git via .gitignore)
├── notebooks/
│   ├── 01_data_collection_and_compilation.ipynb
│   ├── 02_preprocessing_and_dataset_splitting.ipynb
│   ├── 03_corpus_trends_and_linguistic_stats.ipynb
│   ├── 04_model_training_and_evaluation.ipynb
│   └── 05_semantic_field_and_frame_extraction.ipynb
├── src/arabic_news/
│   ├── __init__.py
│   ├── acquisition.py                        # Scraper 6 portal berita, kompilasi teks, & statistik dinamis
│   ├── config.py                             # Path relatif, peta label 11 kelas, metadata portal, & seed
│   ├── evaluation.py                         # Split 80/15/5 & 85/10/5, metrik evaluasi, monitoring psutil
│   ├── features.py                           # Vektorisasi TF-IDF & ekstraksi bobot n-gram per kategori
│   ├── models.py                             # 5 algoritma klasifikasi, inferensi ganda SVM, & skor Sigmoid
│   ├── preprocessing.py                      # Cleaning Arab/ASCII, stopwords, & pembentuk teks RTL
│   └── visualization.py                      # Pencarian font Arab lintas-platform untuk plot & WordCloud
├── tests/
│   └── test_reproducibility.py               # Pengujian otomatis pipeline, notebook, & aplikasi Streamlit
├── .gitignore
├── LICENSE
├── pyproject.toml
└── requirements.txt
```

*(Berkas korpus `data/basic_articles_dataset.csv` berukuran besar dan artefak `.pkl` mentah multi-gigabyte dikecualikan melalui `.gitignore` demi memenuhi batas ukuran GitHub serta menjaga hak cipta portal berita asal. Seluruh keluaran pada direktori `results/` (`stats/`, `lexicons/`, `metrics/`, `figures/`) dan `models/` (`svm_medan_makna.joblib`, `svm_frame_semantik.joblib`) dihasilkan secara dinamis dari sumber tunggal `data/basic_articles_dataset.csv` dan `data/stopwords.txt` saat menjalankan kelima notebook secara berurutan.)*

---

## Instalasi

Direkomendasikan menggunakan Python `3.9`–`3.11` pada *virtual environment* baru:

```bash
git clone <url-repositori-ini>
cd arabic_news_classification

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install --no-deps -e .
```

Letakkan dataset korpus utama `basic_articles_dataset.csv` ke dalam folder `data/` sebelum mengeksekusi notebook (`data/stopwords.txt` telah tersedia di repositori).

---

## Cara Menjalankan Notebook & Pengujian

Kelima notebook disusun berurutan sesuai tahapan metodologi penelitian dan memproses langsung dari `data/basic_articles_dataset.csv` serta `data/stopwords.txt`:

1. `notebooks/01_data_collection_and_compilation.ipynb` — Spesifikasi *web scraper* 6 portal berita Arab, simulasi parsing HTML ke `.txt`, dan komputasi distribusi awal korpus (`results/stats/` & `results/figures/`).
2. `notebooks/02_preprocessing_and_dataset_splitting.ipynb` — Pembersihan teks Arab, percabangan *stopword removal* (Medan Makna) vs retensi preposisi (Semantik *Frame*), pembagian data deterministik (`random_state=42`), serta pemilihan 11 artikel sampel uji ke `results/stats/sample_11_test_articles.csv`.
3. `notebooks/03_corpus_trends_and_linguistic_stats.ipynb` — Komputasi total token sebelum/sesudah *stopword removal*, tipe kosakata unik Arab vs Non-Arab (`results/stats/corpus_summary.json`), serta Top-1000 *Unigram*, *Bigram*, dan *Trigram* tiap kategori (`results/lexicons/`).
4. `notebooks/04_model_training_and_evaluation.ipynb` — Pelatihan, pemantauan CPU/RAM via `psutil`, dan evaluasi penuh 5 algoritma klasifikasi untuk kedua skema (`results/metrics/`), grafik perbandingan & *Confusion Matrix* (`results/figures/`), serta penyimpanan model terbaik SVM terlatih ke `models/`.
5. `notebooks/05_semantic_field_and_frame_extraction.ipynb` — Pembentukan kamus bobot TF-IDF Semantik *Frame* (`results/lexicons/tfidf_top1000/`), ekstraksi ganda Medan Makna & Semantik *Frame* pada 11 artikel uji menggunakan model SVM hasil Notebook 04, analisis koefisien SVM & skor *Sigmoid*, serta visualisasi *WordCloud*.

Untuk memverifikasi seluruh komponen pipeline, notebook, dan aplikasi Streamlit secara otomatis:

```bash
pytest -v
```

---

## Cara Menjalankan Aplikasi Demo Streamlit

Setelah notebook `01`–`05` dijalankan (sehingga `models/` dan `results/` terbentuk), jalankan antarmuka interaktif secara lokal dari root repositori:

```bash
streamlit run app/streamlit_app.py
```

Fitur utama aplikasi Streamlit:
- **Klasifikasi & Ekstraksi Ganda (RTL)**: Masukkan artikel berita Arab sendiri atau pilih contoh dari 11 artikel uji, lalu lihat prediksi kategori secara bersamaan dari model **Semantik *Frame* (`2,3`)** dan **Medan Makna (`1,1`)**, distribusi probabilitas *Sigmoid* 11 kelas, serta sorotan teks RTL berbahasa Arab untuk kata berkoefisien SVM positif (`+`) dan negatif (`-`).
- **Inspeksi Bobot TF-IDF & Koefisien Hyperplane SVM**: Menampilkan daftar kata Medan Makna dan frasa Semantik *Frame* beserta bobot TF-IDF, koefisien `coef_` SVM, dan kontribusinya.
- **Penjelajah Korpus & Kamus 11 Kategori**: Grafik sebaran artikel/portal serta Top-15 frasa *Frame*, *Unigram*, *Bigram*, dan *Trigram* di setiap kategori.
- **Dasbor Evaluasi 5 Algoritma**: Tabel & grafik komparasi akurasi, waktu evaluasi, penggunaan RAM, CPU, dan *heatmap Confusion Matrix* 11×11.

---

## Ringkasan Hasil Evaluasi

### 1. Perbandingan Efektivitas & Efisiensi 5 Model (Korpus Penuh)

| Model | Acc Frame `(2,3)` | F1 Frame | Acc Medan `(1,1)` | F1 Medan | Waktu Frame / Medan (s) | RAM Frame / Medan (MB) |
|---|---:|---:|---:|---:|---:|---:|
| **SVM (`LinearSVC C=10`)** | **`0,9061`** | **`0,9068`** | `0,9026` | `0,9027` | `6,3101` / `0,1259` | `3804,95` / `723,73` |
| **Logistic Regression** | `0,9024` | `0,9033` | **`0,9039`** | **`0,9041`** | `12,0157` / `0,1127` | `4014,11` / `839,48` |
| **Naïve Bayes** | `0,8834` | `0,8835` | `0,8478` | `0,8487` | `11,7019` / `0,1189` | `3587,86` / `112,77` |
| **Random Forest** | `0,8662` | `0,8673` | `0,8684` | `0,8688` | `15,8426` / `2,2433` | `3758,16` / `1323,28` |
| **Gradient Boosting** | `0,8594` | `0,8620` | `0,8810` | `0,8814` | `3,4175` / `3,9849` | `2247,42` / `949,67` |

*(Catatan: Pada pengujian publikasi jurnal awal berbasis Medan Makna Unigram dengan korpus `152.793` artikel, model SVM mencapai akurasi `0,9047`, `Precision = 0,9054`, `Recall = 0,9047`, dan `F1-Score = 0,9049`.)*

### 2. Performa Per-Kategori Model Terbaik SVM (`LinearSVC C=10`, Semantik *Frame*)

| Kategori | Precision | Recall | F1-Score | Support |
|---|---:|---:|---:|---:|
| `أمن` (Keamanan) | `0,9563` | `0,9475` | `0,9519` | `1.734` |
| `اقتصاد` (Ekonomi) | `0,9297` | `0,9397` | `0,9347` | `1.591` |
| `تكنولوجيا` (Teknologi) | `0,8951` | `0,9665` | `0,9294` | `1.104` |
| `ثقافة` (Budaya) | `0,9697` | `0,8848` | `0,9253` | `1.085` |
| `دوليات` (Internasional) | `0,9075` | `0,8898` | `0,8986` | `1.797` |
| `رياضة` (Olahraga) | `0,9861` | `0,9898` | **`0,9879`** | `1.861` |
| `سياحة` (Travel / Pariwisata) | `0,9259` | `0,9304` | `0,9282` | `1.826` |
| `سياسة` (Politik) | `0,9644` | `0,9328` | `0,9483` | `1.829` |
| `صحة` (Kesehatan) | `0,7791` | `0,7421` | `0,7601` | `1.326` |
| `علم` (Sains) | `0,7040` | `0,8092` | `0,7529` | `1.420` |
| `مجتمع` (Masyarakat) | `0,9232` | `0,8844` | `0,9034` | `1.523` |

---

## Sitasi (*Citation*)

Apabila Anda menggunakan kode, metode, atau hasil penelitian dari repositori ini dalam karya ilmiah Anda, silakan merujuk pada kedua publikasi berikut:

1. **Artikel Jurnal (*Studies in English Language and Education*, 2024)**:
   Ahmad, M. I., & Anwari, M. K. (2024). Computational linguistics and natural language processing techniques for semantic field extraction in Arabic online news. *Studies in English Language and Education*, 11(3), 1686–1709. https://doi.org/10.24815/siele.v11i3.38090

2. **Tesis Magister (UIN Sunan Kalijaga Yogyakarta, 2024)**:
   Ahmad, M. I. (2024). *Komputasi Klasifikasi pada Berita Online Berbahasa Arab dengan Pendekatan Semantik Berbasis Frame* [Tesis Magister, Universitas Islam Negeri Sunan Kalijaga Yogyakarta]. https://digilib.uin-suka.ac.id/id/eprint/75147/

```bibtex
@article{ahmad2024computational,
  author  = {Ahmad, Maulana Ihsan and Anwari, Moh. Kanif},
  title   = {Computational Linguistics and Natural Language Processing Techniques for Semantic Field Extraction in Arabic Online News},
  journal = {Studies in English Language and Education},
  year    = {2024},
  volume  = {11},
  number  = {3},
  pages   = {1686--1709},
  doi     = {10.24815/siele.v11i3.38090},
  url     = {https://doi.org/10.24815/siele.v11i3.38090}
}

@mastersthesis{ahmad2024komputasi,
  author = {Ahmad, Maulana Ihsan},
  title  = {Komputasi Klasifikasi pada Berita Online Berbahasa Arab dengan Pendekatan Semantik Berbasis Frame},
  school = {Universitas Islam Negeri Sunan Kalijaga Yogyakarta},
  year   = {2024},
  url    = {https://digilib.uin-suka.ac.id/id/eprint/75147/}
}
```

---

## Lisensi & Kontak

- **Lisensi**: Dirilis di bawah [MIT License](LICENSE).
- **Penulis**: **Maulana Ihsan Ahmad** — Program Studi Magister Bahasa dan Sastra Arab, Fakultas Adab dan Ilmu Budaya, UIN Sunan Kalijaga Yogyakarta.
- **Email**: `23201011009@student.uin-suka.ac.id`
