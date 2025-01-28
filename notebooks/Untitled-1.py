# %%
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

project_root = Path.cwd().parent
sys.path.append(str(project_root))

from config.config import Config
from utils.helper_functions import create_X_y
from utils.helper_fns_for_optuna import load_optimized_model
from models.baseline.baseline_cnn import BinaryCNN, AnimalDataset


# %%
def cross_validate_model(model_name, animal_class, k_folds=5):
    """
    Perform k-fold cross-validation on a specified model and animal class
    """
    # Load data
    print(f"Loading data for {animal_class[0]}...")
    X, y = create_X_y(animal_class)
    
    # Initialize K-Fold
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=Config.RANDOM_SEED)
    
    # Initialize lists to store metrics
    fold_accuracies = []
    fold_losses = []
    
    print(f"Starting {k_folds}-fold cross-validation...")
    
    # Iterate through folds
    for fold, (train_ids, val_ids) in enumerate(kfold.split(X)):
        print(f"\nFold {fold + 1}/{k_folds}")
        
        # Create data loaders for this fold
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        
        # Create datasets
        dataset = AnimalDataset(X, y)
        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)
        optuna = False
        # Load model for this fold
        if 'optuna' in model_name:
            model, _ = load_optimized_model(model_name)
            optuna = True
        else:
            model = BinaryCNN().to(Config.device)
            model.load_state_dict(torch.load(f"{Config.WEIGHTS_DIR}/{model_name}_model.pth"))
        
        # Evaluate model
        model.eval()
        correct = 0
        total = 0
        running_loss = 0
        criterion = nn.BCELoss()
        
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(Config.device)
                
                if optuna:
                    labels = labels.unsqueeze(1)
                labels = labels.to(Config.device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Ensure outputs are within valid range
                outputs = torch.clamp(outputs, 0, 1)
                
                loss = criterion(outputs, labels)
                
                # Calculate accuracy
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
        
        # Calculate metrics for this fold
        fold_accuracy = 100 * correct / total
        fold_loss = running_loss / len(val_loader)
        
        fold_accuracies.append(fold_accuracy)
        fold_losses.append(fold_loss)
        
        print(f"Fold {fold + 1} - Accuracy: {fold_accuracy:.2f}%, Loss: {fold_loss:.4f}")
    
    # Calculate and print summary statistics
    mean_accuracy = np.mean(fold_accuracies)
    std_accuracy = np.std(fold_accuracies)
    
    print(f"\nCross-validation complete!")
    print(f"Mean Accuracy: {mean_accuracy:.2f}% (±{std_accuracy:.2f})")
    
    return fold_accuracies, fold_losses

# %%
def evaluate_model_on_other_classes(model_name, all_classes):
    """
    Evaluate a model on all animal classes and create a performance matrix
    """
    results = {}
    optuna = False
    # Load the model
    if 'optuna' in model_name:
        model, _ = load_optimized_model(model_name)
        optuna = True
    else:
        model = BinaryCNN().to(Config.device)
        model.load_state_dict(torch.load(f"{Config.WEIGHTS_DIR}/{model_name}_model.pth"))
    
    model.eval()
    
    # Test on each class
    for animal_class in all_classes:
        animal_name = animal_class[0]
        print(f"\nEvaluating on {animal_name} dataset...")
        
        # Load data for this class
        X, y = create_X_y(animal_class)
        dataset = AnimalDataset(X, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Evaluate
        correct = 0
        total = 0
        running_loss = 0
        criterion = nn.BCELoss()
        
        with torch.inference_mode():
            for inputs, labels in loader:
                inputs = inputs.to(Config.device)
                
                if optuna:
                    labels = labels.unsqueeze(1)
                labels = labels.to(Config.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                running_loss += loss.item()
        
        # Calculate metrics
        accuracy = 100 * correct / total
        avg_loss = running_loss / len(loader)
        
        results[animal_name] = {
            'accuracy': accuracy,
            'loss': avg_loss
        }
        
        print(f"Accuracy on {animal_name}: {accuracy:.2f}%")
        print(f"Loss on {animal_name}: {avg_loss:.4f}")
    
    return results

# %%
def main():
    # Define all classes
    tiger = ['tiger', 'Tiger_negative_class']
    elephant = ['elephant', 'Elephant_negative_class']
    fox = ['fox', 'Fox_negative_class']
    all_classes = [tiger, elephant, fox]
    
    # Define all models to evaluate
    models = [
        'tiger_baseline',
        'elephant_baseline',
        'fox_baseline',
        'tiger_optuna_final',
        'elephant_optuna_final',
        'fox_optuna_final'
    ]
    
    # Store all results
    cv_results = {}
    cross_class_results = {}
    
    # Perform cross-validation for each model
    print("Performing cross-validation...")
    for model_name in models:
        animal_name = model_name.split('_')[0]
        animal_class = next(c for c in all_classes if c[0] == animal_name)
        
        print(f"\nCross-validating {model_name}...")
        accuracies, losses = cross_validate_model(model_name, animal_class)
        cv_results[model_name] = {
            'accuracies': accuracies,
            'losses': losses
        }
    
    # Evaluate each model on other classes
    print("\nEvaluating models on other classes...")
    for model_name in models:
        print(f"\nEvaluating {model_name}...")
        results = evaluate_model_on_other_classes(model_name, all_classes)
        cross_class_results[model_name] = results
    
    return cv_results, cross_class_results

# %%
def visualize_results(cv_results, cross_class_results):
    # Plot cross-validation results
    plt.figure(figsize=(15, 5))
    
    # Box plot of accuracies
    plt.subplot(1, 2, 1)
    accuracies_data = [cv_results[model]['accuracies'] for model in cv_results]
    plt.boxplot(accuracies_data, labels=[m.replace('_', '\n') for m in cv_results.keys()])
    plt.title('Cross-validation Accuracies')
    plt.ylabel('Accuracy (%)')
    plt.xticks(rotation=45)
    
    # Heatmap of cross-class performance
    plt.subplot(1, 2, 2)
    models = list(cross_class_results.keys())
    animals = ['tiger', 'elephant', 'fox']
    
    performance_matrix = np.zeros((len(models), len(animals)))
    for i, model in enumerate(models):
        for j, animal in enumerate(animals):
            performance_matrix[i, j] = cross_class_results[model][animal]['accuracy']
    
    plt.imshow(performance_matrix, cmap='YlOrRd')
    plt.colorbar(label='Accuracy (%)')
    plt.xticks(range(len(animals)), animals)
    plt.yticks(range(len(models)), [m.replace('_', '\n') for m in models])
    plt.title('Cross-class Performance')
    
    plt.tight_layout()
    plt.show()


# %%
cv_results, cross_class_results = main()

# %%
visualize_results(cv_results, cross_class_results)

# %%



