"""
This script handles the training of the models, including hyperparameter tuning.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def train_model(model, params, X_train_path='data/X_train.csv', y_train_path='data/y_train.csv'):
    """
    Trains a model using GridSearchCV for hyperparameter tuning.

    Args:
        model: The machine learning model to train.
        params (dict): The hyperparameter grid for GridSearchCV.
        X_train_path (str): Path to the training data.
        y_train_path (str): Path to the training labels.

    Returns:
        The best estimator found by GridSearchCV.
    """
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path).values.ravel()

    # Create a pipeline with a scaler and the model
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    # Create the GridSearchCV object
    grid_search = GridSearchCV(pipeline, params, cv=5, n_jobs=-1, scoring='accuracy')

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    print(f"Best score for {model.__class__.__name__}: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")

    return grid_search
