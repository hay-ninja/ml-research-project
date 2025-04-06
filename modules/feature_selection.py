from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from utils import remove_stop_words  # Correctly importing remove_stop_words

def chi_square_selection(X, y, k=100):
    """
    Perform chi-square feature selection.
    Args:
        X: Feature matrix.
        y: Target labels.
        k: Number of top features to select.
    Returns:
        X_new: Transformed feature matrix with top k features.
    """
    selector = SelectKBest(score_func=chi2, k=k)
    X_new = selector.fit_transform(X, y)
    return X_new

def semantic_filtering(X, threshold=0.5):
    """
    Perform semantic filtering.
    Args:
        X: Feature matrix.
        threshold: Similarity threshold for filtering.
    Returns:
        X_filtered: Filtered feature matrix.
    """
    # Placeholder logic: Return the input data as-is
    # Replace this with actual semantic filtering logic if needed
    return X

def pca_selection(X, n_components=100):
    """
    Perform Principal Component Analysis (PCA) for feature selection.
    Args:
        X: Feature matrix.
        n_components: Number of principal components to retain.
    Returns:
        X_new: Transformed feature matrix with selected components.
    """
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(X)
    return X_new

def stop_word_removal(X):
    """
    Perform stop word removal as a feature selection method.
    Args:
        X: Feature matrix (text data).
    Returns:
        X_filtered: Feature matrix with stop words removed.
    """
    # Use remove_stop_words from utils
    X_filtered = remove_stop_words(X)
    return X_filtered
