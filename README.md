# 🤖 Transformer Sentiment Analysis using DistilBERT

Project ini merupakan implementasi sederhana **Transformer** menggunakan model **DistilBERT** dari Hugging Face untuk melakukan **Sentiment Analysis** pada kalimat berbahasa Inggris.

Project dibuat sebagai media pembelajaran **Natural Language Processing (NLP)** untuk memahami penggunaan **Transformer**, khususnya model **DistilBERT**, tanpa harus melakukan proses training dari awal (from scratch).

---

# 🎯 Tujuan Project

Project ini bertujuan untuk:

- Memahami konsep dasar Transformer.
- Menggunakan model **pre-trained DistilBERT**.
- Melakukan klasifikasi sentimen.
- Menampilkan Confidence Score.
- Menyimpan hasil prediksi ke dalam file CSV.
- Melakukan prediksi secara interaktif melalui terminal.

---

# 📂 Struktur Project

```
transformer-demo/
│
├── output/
│   └── transformer_result.csv
│
├── main.py
├── requirements.txt
└── README.md
```

---

# 📚 Dataset

Project ini menggunakan beberapa contoh kalimat (dummy dataset).

Contoh:

| Text | Expected Sentiment |
|------|--------------------|
| This movie is amazing and I really enjoyed it. | Positive |
| The film was terrible and very boring. | Negative |
| The acting was fantastic. | Positive |
| I will never watch this movie again. | Negative |
| The story was interesting and emotional. | Positive |

Dataset dapat diganti dengan:

- IMDb Movie Reviews
- Amazon Reviews
- Twitter Sentiment
- Yelp Reviews
- Custom Dataset

---

# ⚙️ Instalasi

Install seluruh library:

```bash
pip install -r requirements.txt
```

atau

```bash
pip install transformers torch pandas
```

---

# ▶️ Menjalankan Program

Masuk ke folder project

```bash
cd transformer-demo
```

Jalankan program

```bash
python main.py
```

Saat pertama kali dijalankan, model DistilBERT akan diunduh secara otomatis dari Hugging Face. Setelah itu, model akan disimpan dalam cache sehingga eksekusi berikutnya lebih cepat.

---

# 🔄 Workflow

```
Input Text
      │
      ▼
Tokenizer
      │
      ▼
Embedding
      │
      ▼
Transformer Encoder
      │
      ▼
Self-Attention
      │
      ▼
Classification Head
      │
      ▼
Prediction
```

---

# 🧠 Arsitektur Transformer

```
Sentence
    │
    ▼
Tokenizer
    │
    ▼
Embedding
    │
    ▼
Positional Encoding
    │
    ▼
Transformer Encoder
    │
    ▼
Multi-Head Self Attention
    │
    ▼
Feed Forward Network
    │
    ▼
Classification Layer
    │
    ▼
Positive / Negative
```

---

# 📊 Contoh Output

```
============================================================

Loading Transformer Model...

============================================================

Model berhasil dimuat!

============================================================

HASIL PREDIKSI

============================================================

Text :

This movie is amazing and I really enjoyed it.

Prediction :

POSITIVE

Confidence :

0.9998

------------------------------------------------------------

Text :

The film was terrible and very boring.

Prediction :

NEGATIVE

Confidence :

0.9997
```

---

# 📄 Output CSV

Program akan menghasilkan file:

```
output/

transformer_result.csv
```

Contoh isi:

| Text | Prediction | Confidence |
|------|------------|-----------:|
| This movie is amazing... | POSITIVE | 0.9998 |
| The film was terrible... | NEGATIVE | 0.9997 |
| The acting was fantastic. | POSITIVE | 0.9996 |

---

# 📦 Library

Project menggunakan:

- Transformers
- PyTorch
- Pandas

---

# 📖 Istilah Penting

## Transformer

Arsitektur deep learning yang menggunakan mekanisme **Self-Attention** untuk memahami hubungan antar kata dalam suatu kalimat.

---

## DistilBERT

Versi ringan dari BERT yang lebih cepat dan lebih kecil, tetapi tetap mempertahankan sebagian besar performa model asli.

---

## Tokenizer

Mengubah teks menjadi token yang dapat diproses oleh model.

---

## Self-Attention

Mekanisme yang memungkinkan model menentukan kata mana yang paling relevan dengan kata lain dalam satu kalimat.

---

## Confidence Score

Nilai probabilitas yang menunjukkan tingkat keyakinan model terhadap hasil prediksi.

---

# 🚀 Pengembangan Selanjutnya

Project ini dapat dikembangkan menjadi:

- Fine-Tuning BERT pada dataset IMDb.
- Fine-Tuning RoBERTa.
- Multi-Class Text Classification.
- Emotion Detection.
- Fake News Detection.
- Question Answering.
- Text Summarization.
- Translation.
- Streamlit Dashboard.

---

# 📚 Referensi

1. Vaswani, A., et al. (2017). *Attention Is All You Need*.

2. Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*.

3. Sanh, V., et al. (2019). *DistilBERT: A Distilled Version of BERT*.

4. Hugging Face Transformers Documentation

https://huggingface.co/docs/transformers

---

# 👨‍💻 Author

**Arif**

Master's Student in Information Technology

Bidang minat:

- Artificial Intelligence
- Deep Learning
- Natural Language Processing
- Machine Learning
- Data Science

---

# ⭐ Jika project ini bermanfaat

Silakan berikan ⭐ pada repository ini.