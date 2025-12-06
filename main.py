"""
Main script to run the machine learning workflow.
"""
import joblib
import setup
from training import train_model
from models import get_models_and_parameters
from testing import evaluate_model
import os

def main():
    """
    Orchestrates the entire machine learning workflow.
    """
    # Step 1: Get the data
    setup.get_data()

    # Step 2: Get models and parameters
    models_and_params = get_models_and_parameters()

    best_model = None
    best_score = 0
    best_model_name = ""

    results = {}

    # Step 3: Train and evaluate each model
    for model_name, config in models_and_params.items():
        print(f"--- Training {model_name} ---")
        grid_search = train_model(config['model'], config['params'])
        
        # Evaluate model on test set
        metrics = evaluate_model(grid_search.best_estimator_)
        results[model_name] = metrics
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_model = grid_search.best_estimator_
            best_model_name = model_name

    # Step 4: Save the best model
    if best_model:
        if not os.path.exists('trained_models'):
            os.makedirs('trained_models')
        joblib.dump(best_model, f'trained_models/best_{best_model_name.replace(" ", "_").lower()}_model.pkl')
        print(f"\nBest model ({best_model_name}) saved to trained_models/best_{best_model_name.replace(' ', '_').lower()}_model.pkl")

    return results

if __name__ == '__main__':
    main()
