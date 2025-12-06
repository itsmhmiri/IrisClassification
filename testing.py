from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test_path='data/X_test.csv', y_test_path='data/y_test.csv'):
    """
    Evaluates a trained model and returns a dictionary of metrics.
    """
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()
    
    y_pred = model.predict(X_test)
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall': recall_score(y_test, y_pred, average='macro'),
        'f1_score': f1_score(y_test, y_pred, average='macro'),
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }
    
    return metrics

def plot_confusion_matrix(cm, labels, model_name):
    """
    Plots a confusion matrix.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

if __name__ == '__main__':
    # This part is for demonstration and requires a trained model.
    # We will load a model trained with default parameters from models.py
    from models import train_logistic_regression

    # Train a model to use for evaluation
    print("Training a dummy Logistic Regression model for evaluation demonstration...")
    lr_model = train_logistic_regression(random_state=42)
    
    print("\nEvaluating the model...")
    eval_metrics = evaluate_model(lr_model)
    
    print("\nEvaluation Metrics:")
    print(f"  Accuracy: {eval_metrics['accuracy']:.4f}")
    print(f"  Precision: {eval_metrics['precision']:.4f}")
    print(f"  Recall: {eval_metrics['recall']:.4f}")
    print(f"  F1 Score: {eval_metrics['f1_score']:.4f}")
    
    print("\nPlotting Confusion Matrix...")
    iris_species = ['setosa', 'versicolor', 'virginica']
    plot_confusion_matrix(eval_metrics['confusion_matrix'], iris_species, 'Logistic Regression')
    print("Done.")
