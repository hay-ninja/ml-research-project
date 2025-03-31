# main.py
from utils import load_data, preprocess_data, split_data, tfidf_transform
from modules.classifiers import get_naive_bayes, get_svm, get_mlp, get_logistic_regression
from modules.metrics import evaluate_model

print("ho")
def main():
    print("Successfully running.")
    # Update the path here
    file_path = r"C:\Users\hayre\OneDrive\Documents\vscode\ml_method\data\tweet_emotions.csv"
    
    # Load and preprocess the data
    data = load_data(file_path)
    data, label_encoder = preprocess_data(data)
   
    
    # Split the data
    X_train, X_test, y_train, y_test = split_data(data)

    # Feature extraction - TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vectorizer = tfidf_transform(X_train, X_test)

    # Select classifiers
    classifiers = {
        'Naive Bayes': get_naive_bayes(),
        'SVM': get_svm(),
        'MLP': get_mlp(),
        'Logistic Regression': get_logistic_regression()
    }
    
    # Evaluate classifiers
    for clf_name, clf in classifiers.items():
        print(f"Training and evaluating {clf_name}...")
        clf.fit(X_train_tfidf, y_train)
        y_pred = clf.predict(X_test_tfidf)
        
        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(y_test, y_pred)
        print(f"{clf_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
if __name__ == "__main__":
    main()
