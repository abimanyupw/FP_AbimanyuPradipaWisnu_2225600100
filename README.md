# Final Project Oprec PENSHIP — Kode Training Dasar

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Hesyitena/project-oprec/blob/main/train_colab.ipynb)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?logo=ultralytics)
![Python](https://img.shields.io/badge/Python-3.10+-green?logo=python)

Repository ini berisi kode dasar untuk melatih model object detection **YOLOv8** yang mendeteksi 3 kelas objek: `bottle`, `laptop`, dan `pencil`.

> [!IMPORTANT]
> Kode ini sengaja dibuat **sangat dasar** sebagai titik awal. Peserta **wajib mengembangkan kode ini** (misalnya melalui tuning hyperparameter, augmentasi data, analisis hasil, atau penambahan skrip inferensi). Ketentuan lengkap tercantum pada **Guidebook Final Project Oprec PENSHIP** ([intip.in/GuidebookOprecProgram](https://intip.in/GuidebookOprecProgram)).

---

## Daftar Isi
- [Pilihan Lingkungan Training](#pilihan-lingkungan-training)
- [Struktur Proyek](#struktur-proyek)
- [Membuat Repository Peserta](#membuat-repository-peserta)
- [Persiapan Konfigurasi Roboflow](#persiapan-konfigurasi-roboflow)
- [Cara Menjalankan](#cara-menjalankan)
  - [Opsi A: Google Colab (Direkomendasikan)](#opsi-a-google-colab-direkomendasikan)
  - [Opsi B: Komputer Lokal (Windows / Linux / macOS)](#opsi-b-komputer-lokal-windows--linux--macos)
- [Struktur Output Training](#struktur-output-training)
- [Panduan Pengembangan Kode](#panduan-pengembangan-kode)
- [Cara Pengumpulan](#cara-pengumpulan)
  - [Struktur Repository Peserta](#struktur-repository-peserta)
  - [Langkah Pengumpulan](#langkah-pengumpulan)

---

## Pilihan Lingkungan Training

| File | Lingkungan | Rekomendasi Penggunaan |
| --- | --- | --- |
| [`train_colab.ipynb`](train_colab.ipynb) | Google Colab | **Sangat disarankan** apabila laptop tidak memiliki GPU NVIDIA atau memiliki spesifikasi terbatas. Tersedia GPU T4 gratis. |
| [`train.py`](train.py) + [`requirements.txt`](requirements.txt) | Laptop / PC Lokal | Digunakan apabila laptop memiliki GPU NVIDIA dengan driver yang terpasang. |

---

## Struktur Proyek

```text
project-oprec/
├── .gitattributes      # Konfigurasi normalisasi Git (line endings & text diff)
├── .gitignore          # Daftar file/folder yang diabaikan Git (dataset, runs, file .pt kecuali best.pt)
├── requirements.txt    # Daftar dependensi Python utama (ultralytics, roboflow)
├── train.py            # Skrip training dasar untuk eksekusi di komputer lokal
├── train_colab.ipynb   # Jupyter Notebook training untuk Google Colab
├── TEMPLATE_LAPORAN.md # Kerangka laporan; isinya disalin ke README.md repository peserta
└── README.md           # Dokumentasi dan panduan proyek
```

---

## Membuat Repository Peserta

Pengerjaan tidak dilakukan langsung di repository ini. Setiap peserta wajib membuat repository sendiri dengan langkah berikut:

1. Klik **Use this template → Create a new repository** pada halaman ini.
2. Beri nama repository `FP_<NamaLengkap>_<NRP>` dan pilih visibilitas **Public**.
3. Unduh repository tersebut melalui **Code → Download ZIP**, atau clone menggunakan git (ganti `<username>` dengan username GitHub peserta):

```bash
git clone https://github.com/<username>/FP_<NamaLengkap>_<NRP>.git
cd FP_<NamaLengkap>_<NRP>
```

---

## Persiapan Konfigurasi Roboflow

Sebelum training dijalankan, konfigurasi dataset Roboflow disiapkan dengan langkah berikut:

1. Buka dataset di [Roboflow](https://app.roboflow.com).
2. Catat workspace, project, dan nomor version dari URL dataset (`app.roboflow.com/<workspace>/<project>/<version>`) atau dari menu **Download Dataset → YOLOv8 → Show download code**.
3. Ambil **Roboflow API Key** dari halaman [https://app.roboflow.com/settings/api](https://app.roboflow.com/settings/api).

Nilai variabel konfigurasi disesuaikan di bagian `CONFIGURATION` pada [`train.py`](train.py) atau cell **4. Konfigurasi** di notebook:

```python
ROBOFLOW_WORKSPACE = "workspace-name"  # Nama workspace Roboflow
ROBOFLOW_PROJECT   = "project-name"    # Nama project Roboflow
ROBOFLOW_VERSION   = 1                 # Nomor versi dataset
MODEL              = "yolov8n.pt"      # Pilihan: yolov8n.pt atau yolov8s.pt
```

> [!WARNING]
> **Keamanan Kredensial**: API key **dilarang** dituliskan langsung di file kode (`train.py`) maupun notebook (`train_colab.ipynb`), karena repository peserta berstatus Public. API key disimpan melalui Environment Variable atau Colab Secrets.

---

## Cara Menjalankan

### Opsi A: Google Colab (Direkomendasikan)

1. Buka [https://colab.research.google.com/](https://colab.research.google.com/), pilih tab **Upload**, lalu unggah `train_colab.ipynb` dari repository peserta. Badge **Open in Colab** di atas hanya membuka notebook asli milik panitia, bukan versi yang telah dikembangkan peserta.
2. Aktifkan GPU melalui menu **Runtime → Change runtime type → T4 GPU → Save**.
3. Simpan API key di fitur **Secrets (ikon kunci)** Colab dengan nama `ROBOFLOW_API_KEY`, atau masukkan API key saat diminta oleh notebook.
4. Jalankan seluruh cell secara berurutan (**Shift + Enter**) dari langkah 1 sampai langkah 10.
5. Cell nomor 10 mengompres folder output menjadi `runs.zip` dan mengunduh `best.pt` serta grafik hasil ke komputer lokal. Cell ini wajib dijalankan sebelum sesi Colab berakhir.

---

### Opsi B: Komputer Lokal (Windows / Linux / macOS)

#### 1. Persiapan Virtual Environment

**Windows (PowerShell):**
```powershell
# 1. Buat dan aktifkan virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Apabila laptop memiliki GPU NVIDIA, instal PyTorch versi CUDA terlebih dahulu.
#    GTX 9xx/10xx: cu126. GTX 16xx, RTX 20xx-50xx: cu132 (lihat tabel di guidebook).
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132

# 3. Instal dependensi proyek
pip install -r requirements.txt
```

Apabila aktivasi venv ditolak karena *execution policy*, jalankan terlebih dahulu `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` pada jendela PowerShell yang sama.

**Linux / macOS / WSL (Bash):**
```bash
# 1. Buat dan aktifkan virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Instal dependensi proyek
pip install -r requirements.txt
```

#### 2. Jalankan Training

**Windows (PowerShell):**
```powershell
$env:ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
python train.py
```

**Linux / macOS / WSL (Bash):**
```bash
export ROBOFLOW_API_KEY="API_KEY_ROBOFLOW"
python train.py
```

---

## Struktur Output Training

Hasil training tersimpan di folder `runs/detect/train/` (atau `train2/`, `train3/`, dst.):

- `weights/best.pt`: Bobot model dengan performa validation mAP tertinggi. File ini yang dikumpulkan (disalin ke folder utama repository peserta) dan digunakan untuk pengujian.
- `weights/last.pt`: Bobot model pada epoch terakhir training.
- `results.png`: Grafik metrik performa (loss training/validation, precision, recall, mAP50, mAP50-95).
- `confusion_matrix.png`: Matriks evaluasi prediksi per kelas (`bottle`, `laptop`, `pencil`).
- `results.csv`: Catatan numerik metrik performa per epoch.
- `BoxPR_curve.png`: Kurva precision–recall (pada versi Ultralytics lama bernama `PR_curve.png`).

---

## Panduan Pengembangan Kode

Kode dasar ini sengaja dibuat sederhana dan wajib dikembangkan oleh peserta. Batasan yang tetap berlaku:
- Arsitektur model: **YOLOv8n** atau **YOLOv8s** (pre-trained).
- Resolusi input gambar: **`imgsz=640`**.

### Contoh Arah Pengembangan:
1. **Hyperparameter Tuning**: eksplorasi nilai `epochs`, `batch` (16, 8, 4), `lr0` (learning rate awal), `optimizer` (`AdamW`, `SGD`), dan `patience` (early stopping).
2. **Augmentasi Data**: uji kombinasi parameter augmentasi Ultralytics seperti `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `fliplr`, `mosaic`, atau `mixup`.
3. **Eksperimen Terukur**: pencatatan dan perbandingan beberapa skenario percobaan dalam satu tabel perbandingan mAP.
4. **Skrip Inferensi Demo**: pembuatan skrip inferensi tambahan (`predict.py`) untuk menguji deteksi pada foto baru, file video, atau webcam langsung (dapat digunakan untuk membuat video demo).
5. **Analisis Hasil**: penyajian contoh gambar deteksi, evaluasi kasus *false positive* atau *false negative*, beserta penjelasan cara penanganannya.
6. **Kerapian Kode & Ekspor**: implementasi `argparse` / config YAML untuk konfigurasi modular, atau ekspor model ke format ONNX (`model.export(format="onnx")`).

> Referensi lengkap parameter training Ultralytics: [Ultralytics Train Documentation](https://docs.ultralytics.com/modes/train/)

---

## Cara Pengumpulan

Seluruh pekerjaan dikerjakan dan dikumpulkan di repository peserta hasil **Use this template** (Public), bukan di repository ini.

### Struktur Repository Peserta

```text
FP_<NamaLengkap>_<NRP>/
├── train.py atau train_colab.ipynb   # salah satu, yang telah dikembangkan
├── requirements.txt                  # wajib apabila menggunakan train.py
├── ...                               # kode tambahan (apabila ada)
├── best.pt                           # model terbaik, di folder utama
├── grafik/                           # grafik hasil training
└── README.md                         # laporan (dari TEMPLATE_LAPORAN.md)
```

- **Kode:** cukup salah satu, sesuai cara training yang digunakan: `train.py` (training lokal) atau `train_colab.ipynb` (Google Colab). File kode yang tidak digunakan boleh dihapus. `requirements.txt` wajib disertakan apabila menggunakan `train.py`, termasuk package tambahan yang digunakan.
- **best.pt:** disalin dari `runs/detect/train/weights/best.pt` ke folder utama repository (sejajar dengan `README.md`). Apabila training dijalankan lebih dari sekali, gunakan folder training terakhir (`train2`, `train3`, dst.). File `.gitignore` bawaan sudah mengizinkan `best.pt` di folder utama.
- **grafik/:** berisi `results.png`, `results.csv`, `confusion_matrix.png`, dan `BoxPR_curve.png` dari `runs/detect/train/` (pada versi Ultralytics lama bernama `PR_curve.png`).
- **README.md:** seluruh isinya diganti dengan isi [`TEMPLATE_LAPORAN.md`](TEMPLATE_LAPORAN.md), kemudian setiap bagiannya diisi. File `TEMPLATE_LAPORAN.md` boleh dihapus setelahnya.
- **Tidak diunggah:** dataset, folder `runs/`, video demo, dan API key.

Pengunggahan melalui web menggunakan **Add file → Upload files** pada halaman repository (maks. 25 MB per file; `best.pt` YOLOv8n ±6 MB, YOLOv8s ±22 MB), atau melalui git (`git add`, `git commit`, `git push`).

### Langkah Pengumpulan

1. Lengkapi repository peserta sesuai struktur di atas dan pastikan visibilitasnya **Public**.
2. Unggah video demo (30–60 detik) ke Google Drive dengan akses **Anyone with the link → Viewer**.
3. Pastikan project Roboflow berstatus **Public**, lalu cantumkan Project ID (`workspace/project`), nomor version dataset, dan link project publik di README. Project ID juga diisi di Google Form. Link project publik diperoleh dengan mengganti domain `app.roboflow.com` menjadi `universe.roboflow.com` pada URL project, sehingga formatnya `https://universe.roboflow.com/<workspace>/<project>`.
4. Isi Google Form pengumpulan ([forms.gle/P48CRmWXZo4xFzKNA](https://forms.gle/P48CRmWXZo4xFzKNA)) sebelum **Senin, 26 Oktober 2026, pukul 23.59 WIB** dengan nama lengkap, NRP, link repository GitHub, Project ID Roboflow, link video demo di Google Drive, serta pernyataan bahwa dataset diambil sendiri.
5. Repository tidak boleh diubah (commit atau upload) setelah Google Form terkirim.

Ketentuan lengkap dan daftar periksa tercantum pada **Guidebook Final Project Oprec PENSHIP** ([intip.in/GuidebookOprecProgram](https://intip.in/GuidebookOprecProgram)).
