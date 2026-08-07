
# Session 29 (AIML) Assignment
# Emotions Dataset - Vectorization Techniques


# Import Libraries
import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load Dataset

# If using cleaned dataset
# df = pd.read_csv("cleaned_emotions.csv")

# OR Original train.txt
df = pd.read_csv("train.txt", sep=';', names=['text', 'emotion'])


# Text Cleaning

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['text'] = df['text'].apply(clean_text)

print(df.head())


# Q1. Data Preparation

X = df['text']
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nQ1")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape :", y_test.shape)


# Q2. Bag of Words (Unigram)

bow = CountVectorizer()

X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

print("\nQ2")
print("Training Matrix Shape :", X_train_bow.shape)
print("Testing Matrix Shape  :", X_test_bow.shape)

print("\nFirst 20 Features")
print(bow.get_feature_names_out()[:20])


# Q3. Bag of Words + MultinomialNB

model_bow = MultinomialNB()

model_bow.fit(X_train_bow, y_train)

y_pred_bow = model_bow.predict(X_test_bow)

acc_bow = accuracy_score(y_test, y_pred_bow)

print("\nQ3")
print("BoW Accuracy:", acc_bow)


# Q4. Understanding Vocabulary


print("\nQ4")

vocab = bow.get_feature_names_out()

print("Vocabulary Size:", len(vocab))

print("\n15 Vocabulary Words")
print(vocab[:15])

sample = X_train.iloc[0]

print("\nSample Document")
print(sample)

sample_vector = bow.transform([sample])

print("\nBoW Vector")
print(sample_vector)


# Q5. N-Grams (Unigram + Bigram)

bigram = CountVectorizer(ngram_range=(1,2))

X_train_bigram = bigram.fit_transform(X_train)
X_test_bigram = bigram.transform(X_test)

print("\nQ5")

print("Bigram Matrix Shape:", X_train_bigram.shape)

print("\nFirst 20 Bigram Features")
print(bigram.get_feature_names_out()[:20])


# Q6. Bigram + MultinomialNB

model_bigram = MultinomialNB()

model_bigram.fit(X_train_bigram, y_train)

y_pred_bigram = model_bigram.predict(X_test_bigram)

acc_bigram = accuracy_score(y_test, y_pred_bigram)

print("\nQ6")

print("Bigram Accuracy:", acc_bigram)

print("\nComparison")
print("Unigram Accuracy :", acc_bow)
print("Bigram Accuracy  :", acc_bigram)


# Q7. TF-IDF Vectorization

tfidf = TfidfVectorizer()

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nQ7")

print("TF-IDF Train Shape:", X_train_tfidf.shape)
print("TF-IDF Test Shape :", X_test_tfidf.shape)

print("\nFirst 15 Features")
print(tfidf.get_feature_names_out()[:15])


# Q8. TF-IDF + MultinomialNB

model_tfidf = MultinomialNB()

model_tfidf.fit(X_train_tfidf, y_train)

y_pred_tfidf = model_tfidf.predict(X_test_tfidf)

acc_tfidf = accuracy_score(y_test, y_pred_tfidf)

print("\nQ8")

print("TF-IDF Accuracy:", acc_tfidf)


# Q9. Comparison Table

comparison = pd.DataFrame({
    "Vectorizer":[
        "Bag of Words (Unigram)",
        "Bag of Words (Unigram + Bigram)",
        "TF-IDF"
    ],
    "Accuracy":[
        acc_bow,
        acc_bigram,
        acc_tfidf
    ]
})

print("\nQ9")
print(comparison)

best = comparison.loc[comparison["Accuracy"].idxmax()]

print("\nObservation")
print("Best Method:", best["Vectorizer"])
print("Accuracy:", best["Accuracy"])

print("""
Generally,
- Bag of Words uses word frequency only.
- Bigrams capture more context but create many more features.
- TF-IDF reduces the importance of common words and often gives better
  classification performance.
""")


# Q10. Mini Project
print("\nQ10")

print("Training completed.")

if acc_tfidf >= acc_bow:
    best_model = model_tfidf
    best_vectorizer = tfidf
    print("Best Model: TF-IDF + MultinomialNB")
else:
    best_model = model_bow
    best_vectorizer = bow
    print("Best Model: Bag of Words + MultinomialNB")

joblib.dump(best_model, "best_model.pkl")
joblib.dump(best_vectorizer, "best_vectorizer.pkl")

print("Model Saved Successfully.")
print("Files Created:")
print("1. best_model.pkl")
print("2. best_vectorizer.pkl")