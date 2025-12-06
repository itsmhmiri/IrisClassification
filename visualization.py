import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_pairplot(data_path='data/X_train.csv', target_path='data/y_train.csv'):
    """
    Plots a pairplot of the training data.
    """
    X_train = pd.read_csv(data_path)
    y_train = pd.read_csv(target_path)
    
    # Reset index to ensure proper concatenation
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    df = pd.concat([X_train, y_train], axis=1)
    
    # Add a temporary column for species names for better legend
    iris_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(iris_species)
    
    sns.pairplot(df, hue='species', diag_kind='kde')
    plt.suptitle('Pair Plot of Iris Dataset', y=1.02)
    plt.show()

def plot_histograms(data_path='data/X_train.csv'):
    """
    Plots histograms of each feature in the training data.
    """
    X_train = pd.read_csv(data_path)
    X_train.hist(figsize=(10, 8), bins=15, edgecolor='black')
    plt.suptitle('Histograms of Iris Features')
    plt.show()

def plot_boxplots(data_path='data/X_train.csv', target_path='data/y_train.csv'):
    """
    Plots box plots of each feature for each target class.
    """
    X_train = pd.read_csv(data_path)
    y_train = pd.read_csv(target_path)
    
    # Reset index to ensure proper concatenation
    X_train = X_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    df = pd.concat([X_train, y_train], axis=1)
    
    # Add a temporary column for species names for better legend
    iris_species = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}
    df['species'] = df['target'].map(iris_species)
    
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(X_train.columns):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x='species', y=feature, data=df)
    plt.suptitle('Box Plots of Iris Features by Species')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == '__main__':
    # Generate plots if the script is run directly
    print("Generating pair plot...")
    plot_pairplot()
    print("Generating histograms...")
    plot_histograms()
    print("Generating box plots...")
    plot_boxplots()
