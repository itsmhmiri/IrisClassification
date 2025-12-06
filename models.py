from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

def train_logistic_regression(X_train_path='data/X_train.csv', y_train_path='data/y_train.csv', **kwargs):
    """
    Trains a Logistic Regression model.
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    model = LogisticRegression(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train_path='data/X_train.csv', y_train_path='data/y_train.csv', **kwargs):
    """
    Trains a K-Nearest Neighbors model.
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    model = KNeighborsClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train_path='data/X_train.csv', y_train_path='data/y_train.csv', **kwargs):
    """
    Trains a Support Vector Machine model.
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    model = SVC(**kwargs)
    model.fit(X_train, y_train)
    return model

def train_decision_tree(X_train_path='data/X_train.csv', y_train_path='data/y_train.csv', **kwargs):
    """
    Trains a Decision Tree model.
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    
    model = DecisionTreeClassifier(**kwargs)
    model.fit(X_train, y_train)
    return model

if __name__ == '__main__':
    # Example of training a model if the script is run directly
    print("Training Logistic Regression model...")
    lr_model = train_logistic_regression(random_state=42)
    print("Logistic Regression model trained:", lr_model)

    print("\nTraining KNN model...")
    knn_model = train_knn(n_neighbors=5)
    print("KNN model trained:", knn_model)

    print("\nTraining SVM model...")
    svm_model = train_svm(kernel='linear', C=1.0, random_state=42)
    print("SVM model trained:", svm_model)

    print("\nTraining Decision Tree model...")
    dt_model = train_decision_tree(max_depth=3, random_state=42)
    print("Decision Tree model trained:", dt_model)
