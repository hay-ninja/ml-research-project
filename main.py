from utils import load_data, preprocess_data, split_data, tfidf_transform, glove_embedding_transform, load_glove_embeddings
from modules.classifiers import get_naive_bayes, get_svm, get_mlp, get_logistic_regression
from modules.metrics import evaluate_model

def main():
    print("Successfully running.")

    file_path = r"C:\Users\hayre\OneDrive\Documents\vscode\ml_method\data\tweet_emotions.csv"
    
    data = load_data(file_path)
    data, label_encoder = preprocess_data(data)

    X_train, X_test, y_train, y_test = split_data(data)

    # TF-IDF Feature Extraction
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_transform(X_train, X_test)

    # Load GloVe embeddings
    glove_file_path = r'C:\Users\hayre\OneDrive\Documents\vscode\glove.6B\glove.6B.100d.txt'
    glove_embeddings = load_glove_embeddings(glove_file_path, embedding_dim=100)

    # GloVe Feature Extraction
    X_train_embed, X_test_embed = glove_embedding_transform(X_train, X_test, glove_embeddings, embedding_dim=100)

    classifiers = {
        'Naive Bayes': get_naive_bayes(),
        'SVM': get_svm(),
        'MLP': get_mlp(),
        'Logistic Regression': get_logistic_regression()
    }

    for feature_selector, (X_train_feat, X_test_feat) in [
        ('TF-IDF', (X_train_tfidf, X_test_tfidf)),
        ('GloVe Embedding', (X_train_embed, X_test_embed))
    ]:
        print(f"\nEvaluating using {feature_selector}...")

        for clf_name, clf in classifiers.items():
            if feature_selector == 'GloVe Embedding' and clf_name == 'Naive Bayes':
                continue

            print(f"\nTraining {clf_name}...")
            clf.fit(X_train_feat, y_train)

            y_pred_all = clf.predict(X_test_feat)
            accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred_all)
            print(f"{feature_selector} - {clf_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

if __name__ == "__main__":
    main()
