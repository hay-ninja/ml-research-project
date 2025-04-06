from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_X_y, check_array
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.cuda as cuda
import gc

# Naive Bayes - good for text classification
def get_naive_bayes():
    return MultinomialNB()

# SVM with balanced class weights - handles multi-class well
def get_svm():
    return SVC(kernel='linear', class_weight='balanced', probability=True)

# PyTorch MLP optimized for emotion classification
class PyTorchMLP(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden_dim=128, output_dim=2, lr=0.001, epochs=10, 
                 batch_size=32, device=None, patience=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.best_loss = float('inf')
        self.epochs_without_improvement = 0

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
            nn.Softmax(dim=1)
        ).to(self.device)

    def _save_checkpoint(self, state, is_best):
        torch.save(state, 'checkpoint.pt')
        if is_best:
            torch.save(state, 'model_best.pt')

    def fit(self, X, y):
        # Clear GPU memory if using CUDA
        if self.device == 'cuda':
            cuda.empty_cache()
            gc.collect()

        # Data validation and conversion
        X, y = check_X_y(X, y)
        if not isinstance(X, np.ndarray):
            raise ValueError("Input X must be a dense numpy array")

        try:
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            y = torch.tensor(LabelEncoder().fit_transform(y), dtype=torch.long).to(self.device)
        except RuntimeError as e:
            print(f"GPU memory error: {e}")
            print("Falling back to CPU...")
            self.device = 'cpu'
            X = torch.tensor(X, dtype=torch.float32)
            y = torch.tensor(LabelEncoder().fit_transform(y), dtype=torch.long)

        self.model = self._build_model()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2)

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            batch_count = 0

            # Training loop with progress tracking
            for i in range(0, len(X), self.batch_size):
                X_batch = X[i:i + self.batch_size]
                y_batch = y[i:i + self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                batch_count += 1

            avg_loss = total_loss / batch_count
            scheduler.step(avg_loss)

            # Early stopping check
            if avg_loss < self.best_loss:
                self.best_loss = avg_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': self.best_loss,
                    'optimizer': optimizer.state_dict(),
                }, True)
            else:
                self.epochs_without_improvement += 1
                if self.epochs_without_improvement >= self.patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

            if epoch % 2 == 0:
                print(f"Epoch {epoch}: loss = {avg_loss:.4f}")

        return self

    def predict(self, X):
        if self.device == 'cuda':
            cuda.empty_cache()
            gc.collect()

        try:
            X = check_array(X)
            X = torch.tensor(X, dtype=torch.float32).to(self.device)
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(X)
                _, predictions = torch.max(outputs, 1)
            return predictions.cpu().numpy()
        except RuntimeError as e:
            print(f"Error during prediction: {e}")
            return None
        finally:
            if self.device == 'cuda':
                cuda.empty_cache()
                gc.collect()

# Logistic Regression with balanced weights
def get_logistic_regression():
    return LogisticRegression(
        max_iter=2000, 
        class_weight='balanced',
        multi_class='multinomial'
    )
