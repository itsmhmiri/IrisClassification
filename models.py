"""
This file defines the machine learning models and their hyperparameter grids for tuning.
"""
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

def get_models_and_parameters():
    """
    Returns a dictionary of models and their corresponding hyperparameter grids.
    """
    models_and_params = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=200),
            'params': {
                'model__C': [0.1, 1, 10, 100],
                'model__solver': ['liblinear', 'lbfgs']
            }
        },
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {
                'model__n_neighbors': range(1, 10),
                'model__weights': ['uniform', 'distance']
            }
        },
        'SVM': {
            'model': SVC(probability=True),
            'params': {
                'model__C': [0.1, 1, 10, 100],
                'model__gamma': [1, 0.1, 0.01, 0.001],
                'model__kernel': ['rbf', 'linear']
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(),
            'params': {
                'model__max_depth': range(1, 10),
                'model__criterion': ['gini', 'entropy']
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__max_depth': [None, 10, 20, 30]
            }
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__learning_rate': [0.01, 0.1, 0.2],
                'model__max_depth': [3, 5, 7]
            }
        }
    }
    return models_and_params
