import time
from modules.metrics import evaluate_model

def evaluate_feature_selection(classifiers, feature_name, X_train, X_test, y_train, y_test, is_pytorch_dense=False):
    """Evaluate classifiers with a specific feature selection method"""
    print(f"\nEvaluating {feature_name}...")
    results = []
    
    for clf_name, clf in classifiers.items():
        print(f"\nTraining {clf_name} with {feature_name}...")
        try:
            # Handle PyTorch MLP separately if dense data is required
            if clf_name == 'PyTorch MLP' and is_pytorch_dense:
                X_train_use = X_train.toarray() if hasattr(X_train, "toarray") else X_train
                X_test_use = X_test.toarray() if hasattr(X_test, "toarray") else X_test
            else:
                X_train_use, X_test_use = X_train, X_test

            # Skip Naive Bayes for PCA due to negative values
            if clf_name == 'Naive Bayes' and feature_name == 'PCA + TF-IDF':
                print(f"Skipping {clf_name} for PCA + TF-IDF as it is not compatible with negative values.")
                continue

            clf.fit(X_train_use, y_train)
            y_pred = clf.predict(X_test_use)
            metrics = evaluate_model(y_test, y_pred)
            
            results.append({
                'feature': feature_name,
                'classifier': clf_name,
                'metrics': metrics
            })
            
            print(f"{feature_name} - {clf_name} - "
                  f"Accuracy: {metrics[0]:.4f}, "
                  f"Precision: {metrics[1]:.4f}, "
                  f"Recall: {metrics[2]:.4f}, "
                  f"F1: {metrics[3]:.4f}")
                  
        except Exception as e:
            print(f"Error evaluating {clf_name} with {feature_name}: {e}")
            
    return results

def initialize_pytorch_mlp(classifiers, input_dim, output_dim):
    """Initialize PyTorch MLP with correct dimensions"""
    from modules.classifiers import PyTorchMLP
    classifiers['PyTorch MLP'] = PyTorchMLP(
        input_dim=input_dim, 
        output_dim=output_dim
    )
    return classifiers
