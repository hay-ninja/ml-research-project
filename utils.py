# utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder

# Load and preprocess data
def load_data(file_path):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """Preprocess data: handle missing values, encode labels, etc."""
    # Ensure no missing values
    df = df.dropna()

    # Encode labels (Emotions)
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
    
    return df, label_encoder

def split_data(df, test_size=0.2):
    """Split the dataset into training and testing sets."""
    X = df['content']  # Text data
    y = df['sentiment']  # Emotion labels
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    return X_train, X_test, y_train, y_test

# Feature Extraction - TF-IDF
def tfidf_transform(X_train, X_test):
    """Convert text data into TF-IDF vectors."""
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Limit to top 5000 terms
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
