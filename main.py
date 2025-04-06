import os
import time
from utils import load_data, preprocess_data, tfidf_transform
from modules.classifiers import get_naive_bayes, get_svm, get_logistic_regression
from modules.feature_selection import chi_square_selection, pca_selection, semantic_filtering, stop_word_removal
from modules.evaluation import evaluate_feature_selection, initialize_pytorch_mlp

def main():
    start_time = time.time()
    print("Successfully running.")

    # File paths for train, validation, and test datasets
    train_path = r"C:\Users\hayre\OneDrive\Documents\vscode\ml-research-project\data\training.csv"
    val_path = r"C:\Users\hayre\OneDrive\Documents\vscode\ml-research-project\data\validation.csv"
    test_path = r"C:\Users\hayre\OneDrive\Documents\vscode\ml-research-project\data\test.csv"

    # Check if files exist
    file_check_start = time.time()
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: File not found - {path}")
            return
    print(f"File check completed in {time.time() - file_check_start:.2f} seconds.")

    # Load datasets
    load_data_start = time.time()
    train_df, val_df, test_df = load_data(train_path, val_path, test_path)
    print(f"Data loading completed in {time.time() - load_data_start:.2f} seconds.")

    # Preprocess datasets
    preprocess_start = time.time()
    train_df = preprocess_data(train_df)
    val_df = preprocess_data(val_df)
    test_df = preprocess_data(test_df)
    print(f"Data preprocessing completed in {time.time() - preprocess_start:.2f} seconds.")

    # Split data into features and labels
    X_train, y_train = train_df['text'], train_df['label']
    X_val, y_val = val_df['text'], val_df['label']
    X_test, y_test = test_df['text'], test_df['label']

    # Stop Word Removal
    X_train_remove_stop_words = stop_word_removal(X_train)
    X_val_remove_stop_words = stop_word_removal(X_val)
    X_test_remove_stop_words = stop_word_removal(X_test)

    # TF-IDF for Stop Word Removal
    X_train_remove_stop_words_tfidf, X_val_remove_stop_words_tfidf, tfidf_vectorizer_stop_words = tfidf_transform(
        X_train_remove_stop_words, X_val_remove_stop_words
    )
    X_test_remove_stop_words_tfidf = tfidf_vectorizer_stop_words.transform(X_test_remove_stop_words)

    # Semantic Filtering (No GloVe embeddings)
    X_train_semantic = semantic_filtering(X_train, threshold=0.5)
    X_val_semantic = semantic_filtering(X_val, threshold=0.5)
    X_test_semantic = semantic_filtering(X_test, threshold=0.5)

    # Ensure semantic filtering outputs are not None
    if X_train_semantic is None or X_val_semantic is None or X_test_semantic is None:
        raise ValueError("Semantic filtering returned None. Ensure the function is implemented correctly.")

    # TF-IDF for Semantic Filtering
    X_train_semantic_tfidf, X_val_semantic_tfidf, tfidf_vectorizer_semantic = tfidf_transform(X_train_semantic, X_val_semantic)
    X_test_semantic_tfidf = tfidf_vectorizer_semantic.transform(X_test_semantic)

    # TF-IDF for Other Feature Selection Methods
    X_train_tfidf, X_val_tfidf, tfidf_vectorizer = tfidf_transform(X_train, X_val)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    # Chi-Square Feature Selection
    X_train_tfidf_chi = chi_square_selection(X_train_tfidf, y_train, k=100)
    X_val_tfidf_chi = chi_square_selection(X_val_tfidf, y_val, k=100)
    X_test_tfidf_chi = chi_square_selection(X_test_tfidf, y_test, k=100)

    # PCA Feature Selection
    X_train_tfidf_pca = pca_selection(X_train_tfidf, n_components=100)
    X_val_tfidf_pca = pca_selection(X_val_tfidf, n_components=100)
    X_test_tfidf_pca = pca_selection(X_test_tfidf, n_components=100)

    # Initialize classifiers
    classifiers = {
        'Naive Bayes': get_naive_bayes(),
        'SVM': get_svm(),
        'PyTorch MLP': None,  # Will be initialized dynamically
        'Logistic Regression': get_logistic_regression()
    }

    # Evaluate all feature selection methods
    all_results = []
    
    # Stop Word Removal + TF-IDF
    classifiers = initialize_pytorch_mlp(classifiers, X_train_remove_stop_words_tfidf.shape[1], len(set(y_train)))
    results = evaluate_feature_selection(
        classifiers, "Stop Word Removal + TF-IDF",
        X_train_remove_stop_words_tfidf, X_test_remove_stop_words_tfidf,
        y_train, y_test, is_pytorch_dense=True
    )
    all_results.extend(results)

    # Semantic Filtering + TF-IDF
    classifiers = initialize_pytorch_mlp(classifiers, X_train_semantic_tfidf.shape[1], len(set(y_train)))
    results = evaluate_feature_selection(
        classifiers, "Semantic Filtering + TF-IDF",
        X_train_semantic_tfidf, X_test_semantic_tfidf,
        y_train, y_test, is_pytorch_dense=True
    )
    all_results.extend(results)

    # Chi-Square + TF-IDF
    classifiers = initialize_pytorch_mlp(classifiers, X_train_tfidf_chi.shape[1], len(set(y_train)))
    results = evaluate_feature_selection(
        classifiers, "Chi-Square + TF-IDF",
        X_train_tfidf_chi, X_test_tfidf_chi,
        y_train, y_test, is_pytorch_dense=True
    )
    all_results.extend(results)

    # PCA + TF-IDF
    classifiers = initialize_pytorch_mlp(classifiers, X_train_tfidf_pca.shape[1], len(set(y_train)))
    results = evaluate_feature_selection(
        classifiers, "PCA + TF-IDF",
        X_train_tfidf_pca, X_test_tfidf_pca,
        y_train, y_test, is_pytorch_dense=True
    )
    all_results.extend(results)

    # Just TF-IDF
    classifiers = initialize_pytorch_mlp(classifiers, X_train_tfidf.shape[1], len(set(y_train)))
    results = evaluate_feature_selection(
        classifiers, "Just TF-IDF",
        X_train_tfidf, X_test_tfidf,
        y_train, y_test, is_pytorch_dense=True
    )
    all_results.extend(results)

    print(f"Total runtime: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
