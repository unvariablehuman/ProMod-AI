# ProMod AI: Intelligent PTM Site Prediction with 1D-CNN

**ProMod AI** adalah platform berbasis Deep Learning yang dirancang untuk mendeteksi situs *Post-Translational Modification* (PTM) pada sekuens protein menggunakan arsitektur **1D Convolutional Neural Network (1D-CNN)**. Proyek ini dikembangkan sebagai solusi komputasional yang efisien untuk mempercepat penemuan obat dan pemahaman mekanisme penyakit kompleks.

---

## 👥 Anggota Kelompok 5
| Nama | NIM |
| :--- | :--- |
| **Aaron Nikolas Tondosaputro** | 2802412881 |
| **Albani Kalam Haq** | 2802498141 |
| **Justin Lysander Setiawan** | 2802418651 |
| **Kristian Novan** | 2802458560 |
| **Nadya Salsabila** | 2802411790 |
| **Sabrina Arfanindia Devi** | 2802448755 |
| **RIBBY AULIA BARLIE** | 2802392886 |
| **HERLINDA ANGELICA TANJAYA** | 2802397754 |

---

## 🧬 Latar Belakang
*Post-Translational Modification* (PTM) adalah perubahan kimiawi protein setelah sintesis yang berfungsi sebagai "sakelar biologis" pengatur fungsi seluler. Mengingat metode eksperimen laboratorium (seperti spektrometri massa) memakan biaya besar dan waktu yang lama, **ProMod AI** hadir menggunakan pendekatan komputasional untuk mengidentifikasi lokasi modifikasi secara cepat dan akurat.

## 🛠️ Metodologi & Arsitektur Model
Model ini menggunakan **1D-CNN** yang sangat efektif untuk data sekuensial protein (rantai asam amino).

### Alur Kerja Model:
1.  **Input Layer**: Menggunakan jendela sekuens (window size) sepanjang 31 asam amino (15-1-15).
2.  **Encoding**: Representasi numerik menggunakan *One-Hot Encoding*.
3.  **Convolutional Layer**: Mendeteksi motif atau pola lokal asam amino.
4.  **Aktivasi ReLU**: Menambahkan non-linearitas untuk menangkap pola biologis kompleks.
5.  **Max Pooling**: Reduksi dimensi dan ekstraksi fitur paling dominan.
6.  **Fully Connected Layer**: Integrasi informasi fitur untuk klasifikasi.
7.  **Sigmoid Output**: Menghasilkan probabilitas (0-1) keberadaan situs PTM.

## 📊 Dataset & Evaluasi
* **Sumber Data**: Dataset diambil dari [dbPTM](https://biomics.lab.nycu.edu.tw/dbPTM/).
* **Strategi Validasi**: Menggunakan *Protein-Level Split* (GroupShuffleSplit) untuk mencegah *data leakage*.
* **Metrik Evaluasi**: 
    * Accuracy & F1-Score.
    * **Matthews Correlation Coefficient (MCC)** sebagai standar emas bioinformatika.
    * Area Under Curve (ROC-AUC & PR-AUC).

## 🚀 Cara Penggunaan (Widget)
Aplikasi ini dilengkapi dengan widget interaktif untuk melakukan pengujian mandiri:
1. Jalankan semua sel di notebook `bio2.ipynb`.
2. Masukkan sekuens protein dalam format satu huruf (contoh: `MSKGEELFTGV...`) ke dalam kotak teks yang tersedia.
3. Klik tombol **Predict**.
4. Model akan memberikan skor probabilitas dan status apakah residu target merupakan situs PTM atau bukan.

## 📂 Struktur File
* `bio2.ipynb`: Notebook utama (Model final, Evaluasi, & Widget).
* `bio.ipynb`: Notebook eksperimen awal dan analisis data (EDA).
* `README.md`: Dokumentasi proyek.

---
**Catatan:** Proyek ini dikembangkan untuk tujuan akademik sebagai bagian dari studi di BINUS University.
