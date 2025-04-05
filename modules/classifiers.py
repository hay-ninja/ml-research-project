from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

# Naive Bayes
def get_naive_bayes():
    return MultinomialNB()

# SVM with balanced class weights
def get_svm():
    return SVC(kernel='linear', class_weight='balanced')

# Multi-Layer Perceptron Classifier with better convergence settings
def get_mlp():
    return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, solver='adam', learning_rate_init=0.001)

# Logistic Regression with balanced class weights
def get_logistic_regression():
    return LogisticRegression(max_iter=1000, class_weight='balanced')
