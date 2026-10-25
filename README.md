# Final Project Oprec PENSHIP - Abimanyu Pradipa Wisnu (2225600100)

## Identitas
- Nama: Abimanyu Pradipa Wisnu
- NRP: 2225600100
- Program studi: STr Teknik Telekomunikasi

## Project Roboflow
- Project ID Roboflow: abimanyus-workspace-lempe/bottle-laptop-pencil, version 8
- Link Roboflow Universe: https://universe.roboflow.com/abimanyus-workspace-lempe/bottle-laptop-pencil

## Dataset
- Jumlah foto asli: 74 foto (tanpa augmentasi). Jumlah label (bounding box) per kelas: bottle 24, laptop 37, pencil 42.
- Metode pengambilan foto: seluruh foto diambil sendiri dengan kamera HP, dengan variasi jarak (dekat dan jauh), sudut (atas, samping, miring), cahaya (terang dan redup), serta latar (meja, lantai, dan lainnya). Sebagian foto memuat lebih dari satu objek. [ISI/SESUAIKAN: tambahkan detail yang sebenarnya, misalnya apakah ada foto tanpa objek]
- Fitur AI labeling Roboflow yang digunakan (apabila ada) dan cara perapiannya: tidak menggunakan fitur AI labeling (Auto Label, Label Assist, maupun Box Prompting). Seluruh bounding box dibuat manual satu per satu.

## Training
- Lingkungan training: Google Colab (GPU T4)
- Model: yolov8n (`yolov8n.pt`), `imgsz=640`
- Dataset: Roboflow version 8 (resize 640x640, tanpa augmentasi di Roboflow)
- Grafik: lihat folder `grafik/`

## Pengembangan Kode
Daftar perubahan terhadap kode dasar, beserta alasan dan pengaruhnya terhadap hasil:

1. **Jumlah epoch dari 50 menjadi 100.** Pada dataset kecil, nilai validasi sangat tidak stabil di awal training (mAP50 sempat hanya 0.01 pada epoch 11) dan baru stabil setelah sekitar epoch 60. Training yang lebih panjang memberi model waktu untuk konvergen. `best.pt` terpilih pada epoch 94.
2. **Early stopping dengan `patience`.** Pada percobaan awal `patience=20`, training sering berhenti dini (misalnya pada epoch 26 dengan epoch terbaik di epoch 6) sehingga model belum belajar cukup. Nilai `patience` kemudian dinaikkan (50, lalu 80) agar training tidak berhenti terlalu cepat. Pada run akhir, `patience=80` tidak sampai memicu berhenti dini.
3. **Augmentasi `hsv_v=0.5`** (nilai bawaan 0.4) untuk memperluas variasi kecerahan, karena foto diambil pada kondisi cahaya berbeda.
4. **Percobaan rotasi `degrees=15`, lalu dikembalikan ke 0.** Rotasi dicoba untuk ketahanan terhadap sudut, tetapi dugaan saya rotasi kurang cocok untuk pensil yang tipis dan panjang sehingga dimatikan pada run akhir. Ini baru dugaan dan tidak diuji terpisah.
5. **Penambahan dataset secara bertahap** (version 1 sampai 8) pada kelas yang lemah, terutama pensil, berdasarkan hasil confusion matrix dan pengujian pada video.
6. **Cell inferensi video** di akhir notebook (`yolo predict model=runs/detect/train/weights/best.pt source=video_demo.mp4 imgsz=640`) untuk mendeteksi ketiga objek pada video dan menghasilkan video demo.

Catatan: karena data, jumlah epoch, dan augmentasi berubah hampir bersamaan, pengaruh masing-masing perubahan tidak dapat dipisahkan secara pasti.

## Hasil Eksperimen
| No | Skenario | Epochs | Batch | Optimizer | mAP50 | mAP50-95 | Keterangan |
|:--:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| 0 | Baseline (kode dasar), version 1, 19 foto | 50 | 16 | auto | 0.663 | 0.364 | Titik acuan. Set validasi hanya 4 foto sehingga nilainya kurang dapat dipercaya |
| 1 | Version 2, ditambah 10 foto pensil | 50 | 16 | auto | 0.424 | 0.321 | Pensil mulai terdeteksi. Set validasi berubah dan lebih sulit |
| 2 | Version 5, `patience=20`, `hsv_v=0.5`, `degrees=15` | 50 (berhenti pada epoch 26) | 16 | auto | 0.455 | 0.348 | `best.pt` berasal dari epoch 6. Nilai validasi tidak stabil |
| 3 | Version 8 (74 foto), `patience=80`, `hsv_v=0.5`, `degrees=0` | 100 | 16 | auto | 0.814 | 0.562 | Run yang dikumpulkan. `best.pt` dari epoch 94 (precision 0.746, recall 0.762) |

Catatan: dataset dan set validasi berbeda pada tiap baris, sehingga perbandingan antar baris tidak sepenuhnya sebanding. Percobaan pada version 7 (`epochs=100`, `patience=50`) berhenti sekitar epoch 52 karena nilai validasi tidak membaik, sehingga tidak dimasukkan ke tabel.

## Analisis dan Kendala
- Analisis penyebab skenario terbaik menghasilkan nilai lebih tinggi: Skenario 3 (version 8) memberikan hasil terbaik. Dugaan saya, penyebab utamanya adalah jumlah data yang lebih banyak (74 foto, dengan penambahan foto pensil dan foto dari sudut atas), ditambah training yang lebih panjang (100 epoch) karena grafik validasi baru stabil setelah epoch sekitar 60. Mematikan rotasi juga mungkin membantu pensil, tetapi tidak diuji terpisah. Pada confusion matrix awal, pensil hampir tidak pernah terdeteksi benar; setelah foto pensil ditambah, pensil mulai dikenali.
- Hasil pada video demo (49 detik, 1453 frame) dengan model akhir: menurut log `yolo predict`, bottle terdeteksi pada 725 frame, laptop pada 937 frame, dan pencil pada 856 frame. Ketiganya muncul bersamaan pada 556 frame, dan 195 frame tidak memiliki deteksi sama sekali. Angka ini menghitung frame yang memuat deteksi, bukan memeriksa kebenarannya satu per satu. Dari cuplikan yang diperiksa manual, botol yang besar dan dekat dengan kamera beberapa kali tidak terdeteksi, dan ada kotak ganda pada pensil di beberapa frame.
- Kelemahan yang masih ada: pada pengujian video dengan model versi awal, laptop yang difoto dari atas tidak terdeteksi, dan muncul false positive `pencil` pada stiker laptop, tepi meja, dan lantai. Pensil yang kecil di dalam frame juga sulit dikenali. Kelas `bottle` memiliki label paling sedikit (24) sehingga botol yang kecil atau terpotong di pinggir frame cenderung terlewat.
- Kendala yang ditemui dan cara penyelesaiannya:
  - Dataset awal sangat sedikit sehingga set validasi hanya 4 foto dan nilai mAP berubah-ubah. Diatasi dengan menambah foto secara bertahap dan membuat version dataset baru.
  - Laptop tidak memiliki GPU NVIDIA, sehingga training dilakukan di Google Colab (GPU T4).
  - API key Roboflow disimpan di Colab Secrets agar tidak tertulis di kode.
  - Folder `dataset/` di Colab harus dihapus setiap berganti version, karena kode dasar melewati proses unduh bila folder tersebut sudah ada.
