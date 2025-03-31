from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
# Add CNN and RNN using a framework like TensorFlow or PyTorch

# Naive Bayes
def get_naive_bayes():
    return MultinomialNB()

# Support Vector Machine
def get_svm():
    return SVC(kernel='linear')

# Multi-Layer Perceptron Classifier
def get_mlp():
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=300)

# Logistic Regression
def get_logistic_regression():
    return LogisticRegression(max_iter=1000)

# Add CNN/RNN methods when ready
