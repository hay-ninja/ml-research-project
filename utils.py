import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import numpy as np

# Load datasets (train, validation, test)
def load_data(train_path, val_path, test_path):
    # Skip the header row to avoid including column names as data
    train_df = pd.read_csv(train_path, header=0, names=['text', 'label'])
    val_df = pd.read_csv(val_path, header=0, names=['text', 'label'])
    test_df = pd.read_csv(test_path, header=0, names=['text', 'label'])
    return train_df, val_df, test_df

# Preprocess data
def preprocess_data(df):
    # Ensure the dataset has the correct columns
    df['text'] = df['text'].astype(str)  # Ensure text is string
    df['label'] = pd.to_numeric(df['label'], errors='coerce')  # Convert label to numeric, coercing errors to NaN
    df = df.dropna(subset=['label'])  # Drop rows with invalid labels
    df['label'] = df['label'].astype(int)  # Ensure label is integer
    return df

# Remove stop words
def remove_stop_words(X):
    """
    Remove stop words from the input text data.
    Args:
        X: List or Series of text data.
    Returns:
        X_filtered: List of text data with stop words removed.
    """
    def filter_stop_words(text):
        return ' '.join([word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS])

    return X.apply(filter_stop_words) if isinstance(X, pd.Series) else [filter_stop_words(text) for text in X]

# Split data with stratification
def split_data(df, test_size=0.2, random_state=42):
    X = df['content']
    y = df['sentiment']
    
    print("Label distribution:\n", y.value_counts())  # Debugging step

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)

# TF-IDF transformation
def tfidf_transform(X_train, X_test):
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, ngram_range=(1,2))
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
