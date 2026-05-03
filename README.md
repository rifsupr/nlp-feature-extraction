# NLP Feature Extraction - Indonesian Abusive & Hate Speech Twitter Dataset

**Natural Language Processing (NLP)**

## Penjelasan Project

Project ini berisi implementasi tahapan **Feature Extraction** (Ekstraksi Fitur) menggunakan teknik Natural Language Processing (NLP) pada dataset teks Twitter berbahasa Indonesia yang sebelumnya telah melalui tahap *preprocessing*. Tujuan dari project ini adalah untuk mengonversi data teks bersih (*clean text*) menjadi representasi angka (vektor numerik) agar dapat diproses oleh algoritma Machine Learning pada tahap pemodelan (*modeling*) selanjutnya.

## Tahapan Ekstraksi Fitur

Proses feature extraction yang dilakukan dalam project ini meliputi dua metode yang paling populer di NLP:

1. **TF-IDF (Term Frequency - Inverse Document Frequency)**:
   Metode ini memberikan bobot pada setiap kata berdasarkan seberapa sering kata tersebut muncul dalam sebuah dokumen (tweet) dibandingkan dengan frekuensinya di seluruh korpus dokumen. Kata yang sering muncul di satu dokumen tapi jarang di dokumen lain akan mendapat bobot tinggi.
2. **Bag of Words (BoW / CountVectorizer)**:
   Metode ini menghitung frekuensi (jumlah kemunculan) setiap kata dalam setiap dokumen (tweet). Berbeda dengan TF-IDF yang memperhitungkan bobot *inverse*, BoW hanya berfokus pada seberapa banyak kata tersebut muncul murni berdasarkan *count*.

Kedua metode ini diimplementasikan menggunakan library `scikit-learn`.

## Library yang Diinstall

Project ini menggunakan bahasa pemrograman **Python 3**. Beberapa library utama yang dibutuhkan dan perlu diinstall meliputi:

- **pandas**: Digunakan untuk membaca dataset CSV.
- **scikit-learn**: Digunakan untuk menyediakan fungsi ekstraksi fitur (menggunakan modul `TfidfVectorizer` dan `CountVectorizer`).
- **scipy**: Digunakan untuk menyimpan matriks fitur yang bersifat renggang (*sparse matrix*) secara efisien menggunakan format `.npz`.
- **numpy**: Mendukung proses komputasi numerik.

Untuk menginstall semua *dependencies*, Anda dapat menjalankan perintah berikut:
```bash
pip install pandas scikit-learn scipy numpy
```

## 📊 Penjelasan Dataset Input

Dataset yang digunakan sebagai input dalam program ini adalah `clean_data.csv`. File ini merupakan hasil ekstraksi dari tahap **Preprocessing** yang dilakukan sebelumnya. 
- Berisi ~13.169 baris teks tweet berbahasa Indonesia yang telah melewati tahap *Case Folding*, *Cleansing*, *Tokenizing*, *Filtering*, dan *Stemming*.
- Target kolom teks yang diekstrak adalah kolom `hasil_preprocessing`.
- Jika terdapat nilai teks kosong (*missing values*) yang terjadi karena proses *cleansing/stemming*, program secara otomatis akan menghapus baris kosong tersebut sebelum memulai ekstraksi.

## 💾 Hasil Ekstraksi (Output)

Proses ini akan menghasilkan 4 buah file output yang akan disimpan di dalam folder `output/`:

1. **`tfidf_features.npz`**: Matriks sparse (berisi angka/bobot numerik) hasil perhitungan metode TF-IDF untuk seluruh korpus data.
2. **`tfidf_vectorizer.pkl`**: Objek model `TfidfVectorizer` yang telah di-*fit* dengan data pelatihan. Sangat berguna untuk di-*load* di kemudian hari ketika ingin mengekstrak teks baru di proses klasifikasi (*deployment*).
3. **`bow_features.npz`**: Matriks sparse (berisi angka/frekuensi kemunculan kata) hasil perhitungan metode Bag of Words.
4. **`bow_vectorizer.pkl`**: Objek model `CountVectorizer` yang telah di-*fit*.

## 🚀 Cara Menjalankan

1. Pastikan Anda telah menyelesaikan tahap *Preprocessing* dan dataset bersih berada di lokasi root proyek: `../datasets/clean_data.csv`.
2. Buka terminal/CMD dan pastikan Anda berada di dalam direktori `feature_extraction`.
3. Jalankan script utama:
   ```bash
   python main.py
   ```
4. Tunggu beberapa saat hingga program selesai memproses belasan ribu data.
5. Hasil akhirnya akan muncul di layar konsol beserta informasi mengenai ukuran *vocabulary* dan dimensi matriks fitur. File akan otomatis tersimpan di folder `output/`.

## 📁 Struktur Project

```text
feature_extraction/
├── output/
│   ├── tfidf_features.npz      # Matriks fitur TF-IDF
│   ├── tfidf_vectorizer.pkl    # Model TfidfVectorizer
│   ├── bow_features.npz        # Matriks fitur BoW
│   └── bow_vectorizer.pkl      # Model CountVectorizer
├── main.py                     # File script utama untuk tahapan feature extraction
└── README.md                   # Dokumentasi project ini
```
