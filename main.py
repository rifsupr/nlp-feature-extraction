import pandas as pd
import numpy as np
import os
import pickle
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def load_data(filepath):
    print("============================================================")
    print(f"LOADING DATASET DARI:\n{filepath}")
    print("============================================================")
    df = pd.read_csv(filepath)
    print(f"Jumlah data: {len(df)} baris")
    return df

def handle_missing_values(df, column_name):
    # Cek nilai NaN/Null
    missing = df[column_name].isnull().sum()
    if missing > 0:
        print(f"Ditemukan {missing} baris dengan nilai kosong. Menghapus baris kosong...")
        df = df.dropna(subset=[column_name])
        print(f"Jumlah data setelah penghapusan: {len(df)} baris")
    
    # Pastikan data type adalah string
    df = df.copy()
    df[column_name] = df[column_name].astype(str)
    return df

def main():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, 'datasets', 'clean_data.csv')
    output_dir = os.path.join(base_dir, 'output')
    
    # Buat direktori output jika belum ada
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Membuat direktori {output_dir}")
        
    # 1. Load Data
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {data_path}")
        print("Pastikan Anda sudah menjalankan script preprocessing dan meletakkan file di lokasi yang benar.")
        return

    text_column = 'hasil_preprocessing'
    if text_column not in df.columns:
        print(f"Error: Kolom '{text_column}' tidak ditemukan dalam dataset.")
        print(f"Kolom yang tersedia: {df.columns.tolist()}")
        return

    # 2. Handle missing values
    df = handle_missing_values(df, text_column)
    texts = df[text_column].values

    # 3. Ekstraksi Fitur: TF-IDF
    print("\n============================================================")
    print("TAHAP 1: EKSTRAKSI FITUR MENGGUNAKAN TF-IDF")
    print("============================================================")
    print("Proses ekstraksi sedang berjalan...")
    
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_features = tfidf_vectorizer.fit_transform(texts)
    
    print(f"Ukuran vocabulary (jumlah kata unik): {len(tfidf_vectorizer.vocabulary_)}")
    print(f"Dimensi matriks TF-IDF (Baris, Kolom): {tfidf_features.shape}")

    # 4. Ekstraksi Fitur: Bag of Words (BoW)
    print("\n============================================================")
    print("TAHAP 2: EKSTRAKSI FITUR MENGGUNAKAN BAG OF WORDS (BoW)")
    print("============================================================")
    print("Proses ekstraksi sedang berjalan...")
    
    bow_vectorizer = CountVectorizer()
    bow_features = bow_vectorizer.fit_transform(texts)
    
    print(f"Ukuran vocabulary (jumlah kata unik): {len(bow_vectorizer.vocabulary_)}")
    print(f"Dimensi matriks BoW (Baris, Kolom): {bow_features.shape}")

    # 5. Menyimpan Hasil Ekstraksi
    print("\n============================================================")
    print("MENYIMPAN HASIL EKSTRAKSI FITUR...")
    print("============================================================")
    
    # Path penyimpananan
    tfidf_matrix_path = os.path.join(output_dir, 'tfidf_features.npz')
    tfidf_model_path = os.path.join(output_dir, 'tfidf_vectorizer.pkl')
    bow_matrix_path = os.path.join(output_dir, 'bow_features.npz')
    bow_model_path = os.path.join(output_dir, 'bow_vectorizer.pkl')
    
    # Save sparse matrices
    sparse.save_npz(tfidf_matrix_path, tfidf_features)
    sparse.save_npz(bow_matrix_path, bow_features)
    
    # Save models
    with open(tfidf_model_path, 'wb') as f:
        pickle.dump(tfidf_vectorizer, f)
    with open(bow_model_path, 'wb') as f:
        pickle.dump(bow_vectorizer, f)
        
    print(f"-> Matriks TF-IDF disimpan di: output/tfidf_features.npz")
    print(f"-> Model TF-IDF disimpan di  : output/tfidf_vectorizer.pkl")
    print(f"-> Matriks BoW disimpan di   : output/bow_features.npz")
    print(f"-> Model BoW disimpan di     : output/bow_vectorizer.pkl")
    
    print("\n✅ Proses Feature Extraction Selesai!")

if __name__ == "__main__":
    main()
