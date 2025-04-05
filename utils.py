import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Function to load GloVe embeddings
def load_glove_embeddings(glove_file_path, embedding_dim=100):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print(f'Loaded {len(embeddings_index)} word vectors from GloVe.')
    return embeddings_index

# Function to get the average GloVe vector
def get_avg_glove_vector(tokens, glove_embeddings, embedding_dim=100):
    vectors = [glove_embeddings.get(token, np.zeros(embedding_dim)) for token in tokens if token in glove_embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(embedding_dim)

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Preprocess data
def preprocess_data(df):
    df = df.drop(columns=['tweet_id'])  # Drop ID column
    label_encoder = LabelEncoder()
    df['sentiment'] = label_encoder.fit_transform(df['sentiment'])
    return df, label_encoder

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

# GloVe embedding transformation
def glove_embedding_transform(X_train, X_test, glove_embeddings, embedding_dim=100):
    X_train_tokens = [text.split() for text in X_train]
    X_test_tokens = [text.split() for text in X_test]
    
    X_train_embed = np.array([get_avg_glove_vector(tokens, glove_embeddings, embedding_dim) for tokens in X_train_tokens])
    X_test_embed = np.array([get_avg_glove_vector(tokens, glove_embeddings, embedding_dim) for tokens in X_test_tokens])
    
    return X_train_embed, X_test_embed
