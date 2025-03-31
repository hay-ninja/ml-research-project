import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

# Function to load GloVe embeddings from a file
def load_glove_embeddings(glove_file_path, embedding_dim=100):
    """Loads GloVe embeddings into a dictionary."""
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Loaded {len(embeddings_index)} word vectors from GloVe.')
    return embeddings_index

# Function to get the average GloVe vector for a given text
def get_avg_glove_vector(tokens, glove_embeddings, embedding_dim=100):
    """Returns the average GloVe vector for a given text."""
    vectors = [glove_embeddings.get(token, np.zeros(embedding_dim)) for token in tokens]
    return np.mean(vectors, axis=0)

# Load data function
def load_data(file_path):
    """Loads CSV file and returns a DataFrame."""
    df = pd.read_csv(file_path)
    return df

# Preprocess data function
def preprocess_data(df):
    """Encodes sentiment labels and removes unnecessary columns."""
    df = df.drop(columns=['tweet_id'])  # Drop tweet_id since it's irrelevant for ML

    # Encode sentiment labels to numbers
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])

    return df, label_encoder

# Split data function
def split_data(df, test_size=0.2, random_state=42):
    """Splits the dataset into training and testing sets."""
    X = df['content']  # Features (tweet text)
    y = df['sentiment']  # Labels (encoded sentiment)

    return train_test_split(X, y, test_size=test_size, random_state=random_state)

# TF-IDF transform function (optional, keeping it in case you still want to use it)
def tfidf_transform(X_train, X_test):
    """Converts text data into TF-IDF feature vectors."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer

# GloVe transform function
def glove_embedding_transform(X_train, X_test, glove_embeddings, embedding_dim=100):
    """Converts text data into GloVe embeddings."""
    # Tokenize text
    X_train_tokens = [text.split() for text in X_train]
    X_test_tokens = [text.split() for text in X_test]
    
    # Transform both training and testing data
    X_train_embed = np.array([get_avg_glove_vector(tokens, glove_embeddings, embedding_dim) for tokens in X_train_tokens])
    X_test_embed = np.array([get_avg_glove_vector(tokens, glove_embeddings, embedding_dim) for tokens in X_test_tokens])
    
    return X_train_embed, X_test_embed
