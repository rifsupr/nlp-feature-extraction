from transformers import pipeline
import pandas as pd

# ======================================================
# LOAD TRANSFORMER MODEL
# ======================================================

print("=" * 60)
print("Loading Transformer Model...")
print("=" * 60)

classifier = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

print("Model berhasil dimuat!\n")

# ======================================================
# DATA UJI
# ======================================================

texts = [

    "This movie is amazing and I really enjoyed it.",

    "The film was terrible and very boring.",

    "The acting was fantastic.",

    "I will never watch this movie again.",

    "The story was interesting and emotional."

]

# ======================================================
# PREDIKSI
# ======================================================

results = []

print("=" * 60)
print("HASIL PREDIKSI")
print("=" * 60)

for text in texts:

    result = classifier(text)[0]

    print(f"\nText      : {text}")
    print(f"Label     : {result['label']}")
    print(f"Confidence: {result['score']:.4f}")

    results.append({

        "Text": text,

        "Prediction": result["label"],

        "Confidence": round(result["score"], 4)

    })

# ======================================================
# TABEL HASIL
# ======================================================

df = pd.DataFrame(results)

print("\n")
print("=" * 60)
print("RINGKASAN")
print("=" * 60)

print(df)

# ======================================================
# SIMPAN CSV
# ======================================================

df.to_csv(

    "output/transformer_result.csv",

    index=False

)

print("\nHasil berhasil disimpan pada output/transformer_result.csv")

# ======================================================
# INPUT USER
# ======================================================

print("\n")
print("=" * 60)
print("PREDIKSI INTERAKTIF")
print("=" * 60)

while True:

    sentence = input("\nMasukkan kalimat (ketik 'exit' untuk keluar): ")

    if sentence.lower() == "exit":
        break

    result = classifier(sentence)[0]

    print(f"\nPrediksi : {result['label']}")
    print(f"Confidence : {result['score']:.4f}")