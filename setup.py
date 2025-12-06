import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import os

def create_features(X):
    """
    Creates new features from the existing ones.
    """
    X['petal area'] = X['petal length (cm)'] * X['petal width (cm)']
    X['sepal area'] = X['sepal length (cm)'] * X['sepal width (cm)']
    return X

def get_data(test_size=0.2, random_state=42):
    """
    Downloads the Iris dataset, creates new features, splits it into training 
    and testing sets, and saves them as CSV files in the 'data' directory.
    """
    # Create the data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Load the Iris dataset
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target, name='target')

    # Create new features
    X = create_features(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save the splits to CSV
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)

    print("Data downloaded, features created, and data split successfully.")
    print("Train and test sets saved in the 'data' directory.")

if __name__ == '__main__':
    get_data()