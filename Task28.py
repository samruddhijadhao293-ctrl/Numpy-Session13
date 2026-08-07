print("Question no.1")
import pandas as pd

# Load dataset
df = pd.read_csv("train.txt", sep=";", names=["text", "emotions"])

# First 10 rows
print("First 10 Rows:")
print(df.head(10))

# Shape
print("\nDataset Shape:")
print(df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

print("Question no.2")
from sklearn.preprocessing import LabelEncoder

# Unique emotions
print("Unique Emotion Labels:")
print(df["emotions"].unique())

# Label Encoding
encoder = LabelEncoder()

df["emotion_encoded"] = encoder.fit_transform(df["emotions"])

# Mapping dictionary
mapping = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

print("\nEmotion Mapping:")
print(mapping)

# Display dataframe
print(df.head())

print("Question no.3")
# Save original text
df["original_text"] = df["text"]

# Convert to lowercase
df["text"] = df["text"].str.lower()

# Before and After
print("Before Lowercase:")
print(df["original_text"].head())

print("\nAfter Lowercase:")
print(df["text"].head())

print("Question no.4")
import string

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))

# Before cleaning
before = df["text"].copy()

# Apply
df["text"] = df["text"].apply(remove_punctuation)

# Display examples
for b, a in zip(before.head(), df["text"].head()):
    print("Before:", b)
    print("After :", a)
    print()

print("Question no.5")    
import re

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

before = df["text"].copy()

df["text"] = df["text"].apply(remove_numbers)

for b, a in zip(before.head(), df["text"].head()):
    print("Before:", b)
    print("After :", a)
    print()

print("Question no.6")
def remove_non_ascii(text):
        return text.encode("ascii", "ignore").decode()

df["text"] = df["text"].apply(remove_non_ascii)

print(df["text"].head())

print("Question no.7")
import nltk

nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def remove_stopwords(text):
    words = word_tokenize(text)
    filtered = [word for word in words if word not in stop_words]
    return " ".join(filtered)

df["text"] = df["text"].apply(remove_stopwords)

print(df["text"].head())

print("Question no.8")
import string
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))

def clean_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Remove emojis/non-ascii
    text = text.encode("ascii", "ignore").decode()

    # Remove stopwords
    words = word_tokenize(text)
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df["cleaned_text"] = df["original_text"].apply(clean_text)

print(df[["original_text", "cleaned_text"]].head(10))

print("Question no.9")
import matplotlib.pyplot as plt

# Word count
df["text_length"] = df["cleaned_text"].apply(lambda x: len(x.split()))

# Statistics
print("Average Length:", df["text_length"].mean())
print("Minimum Length:", df["text_length"].min())
print("Maximum Length:", df["text_length"].max())

# Histogram
plt.figure(figsize=(8,5))
plt.hist(df["text_length"], bins=30, color="skyblue", edgecolor="black")
plt.title("Histogram of Cleaned Text Length")
plt.xlabel("Number of Words")
plt.ylabel("Frequency")
plt.show()

print("Question no.10")
import pandas as pd
import string
import re
import nltk

from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load dataset
df = pd.read_csv("train.txt", sep=";", names=["text", "emotions"])

# Label Encoding
encoder = LabelEncoder()
df["emotion_encoded"] = encoder.fit_transform(df["emotions"])

# Stopwords
stop_words = set(stopwords.words("english"))

# Cleaning function
def clean_text(text):

    text = text.lower()

    text = text.translate(str.maketrans('', '', string.punctuation))

    text = re.sub(r'\d+', '', text)

    text = text.encode("ascii", "ignore").decode()

    words = word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Apply cleaning
df["cleaned_text"] = df["text"].apply(clean_text)

# Save cleaned dataset
df.to_csv("cleaned_emotions.csv", index=False)

print("Cleaned dataset saved successfully.")

# Emotion counts
print("\nEmotion Counts:")
print(df["emotions"].value_counts())
