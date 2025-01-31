import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import optuna
import sys
from pathlib import Path
project_root = Path.cwd()
sys.path.append(str(project_root))
from config.config import Config
from utils.helper_functions import create_X_y, save_model 
from models.binary_cnn_old.baseline_cnn import AnimalDataset

class OptimizedCNN(nn.Module):
    """CNN architecture with hyperparameters determined by Optuna"""
    def __init__(self, trial):
        super(OptimizedCNN, self).__init__()
        
        n_layers = trial.suggest_int('n_layers', 1, 3)
        n_channels = [3]
        
        # Build channel structure for each layer
        for i in range(n_layers):
            out_channels = trial.suggest_int(f'n_channels_l{i}', 16, 128)
            n_channels.append(out_channels)
        
        self.conv_layers = nn.ModuleList()
        curr_size = Config.IMG_SIZE
        
        for i in range(n_layers):
            dropout = trial.suggest_float(f'dropout_l{i}', 0.1, 0.5)
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(n_channels[i], n_channels[i+1], kernel_size=3, padding=1),
                nn.BatchNorm2d(n_channels[i+1]),
                nn.ReLU(),
                nn.Dropout2d(dropout),
                nn.MaxPool2d(2)
            ))
            curr_size //= 2
        
        # Calculate size for fully connected layers
        flattened_size = n_channels[-1] * curr_size * curr_size
        
        # Build fully connected layers
        n_fc_layers = trial.suggest_int('n_fc_layers', 1, 2)
        self.fc_layers = nn.ModuleList()
        prev_size = flattened_size
        
        for i in range(n_fc_layers):
            fc_size = trial.suggest_int(f'n_fc_units_l{i}', 64, 512)
            fc_dropout = trial.suggest_float(f'fc_dropout_l{i}', 0.1, 0.5)
            
            self.fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ])
            prev_size = fc_size
        
        # Output layer without sigmoid (using BCEWithLogitsLoss)
        self.fc_layers.append(nn.Linear(prev_size, 1))
    
    def forward(self, x):
        # Input validation
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor (batch_size, channels, height, width), got shape {x.shape}")
        
        # Forward pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        
        for fc_layer in self.fc_layers:
            x = fc_layer(x)
        
        return x

def objective(trial, X, y, device):
    """Objective function for Optuna optimization"""
    
    batch_size = trial.suggest_int('batch_size', 32, 64)
    lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log=True)
    
    # Split data
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    train_dataset = AnimalDataset(X_train, y_train)
    val_dataset = AnimalDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model and training components
    model = OptimizedCNN(trial).to(device)
    
    # Calculate class weights for imbalanced dataset
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    n_epochs = trial.suggest_int('n_epochs', 20, 50)
    best_val_accuracy = 0.0
    patience = 5
    patience_counter = 0
    
    try:
        for epoch in range(n_epochs):
            # Training phase
            model.train()
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1)
                
                optimizer.zero_grad(set_to_none=True)  # for memory efficiency
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                with torch.inference_mode():
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    train_total += labels.size(0)
                    train_correct += (predicted == labels).sum().item()
            
            # Validation phase
            model.eval()
            val_correct = 0
            val_total = 0
            
            with torch.inference_mode():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    labels = labels.view(-1, 1)
                    outputs = model(inputs)
                    predicted = (torch.sigmoid(outputs) > 0.5).float()
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            # Calculate validation accuracy
            val_accuracy = 100.0 * val_correct / val_total
            
            # Early stopping check
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
            
            trial.report(val_accuracy, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Clear memory
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"WARNING: out of memory error in trial {trial.number}")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            raise optuna.TrialPruned()
        raise e
    
    return best_val_accuracy

def train_final_model(trial, X, y, device):
    """Train the final model using the best hyperparameters"""
    # Get hyperparameters from the trial
    batch_size = trial.params['batch_size']
    lr = trial.params['lr']
    weight_decay = trial.params['weight_decay']
    n_epochs = trial.params['n_epochs']
    
    # Prepare data
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create data loaders
    train_dataset = AnimalDataset(X_train, y_train)
    val_dataset = AnimalDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = OptimizedCNN(trial).to(device)
    pos_weight = torch.tensor([(y_train == 0).sum() / (y_train == 1).sum()]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            labels = labels.view(-1, 1)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (torch.sigmoid(outputs) > 0.5).float()
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        epoch_train_loss = train_loss / len(train_loader)
        epoch_train_acc = 100 * train_correct / train_total
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                labels = labels.view(-1, 1)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        # Calculate validation metrics
        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total
        val_losses.append(epoch_val_loss)
        val_accuracies.append(epoch_val_acc)
        
        print(f'Epoch [{epoch+1}/{n_epochs}]')
        print(f'Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%')
        print(f'Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}%')
        print('-' * 50)
    
    return model, (train_losses, val_losses, train_accuracies, val_accuracies)

def optimize_model(X, y, animal_name, n_trials=100):
    """Run Optuna optimization and train final model"""
    device = torch.device(Config.device)
    
    # Create study object
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_SEED)
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X, y, device),
        n_trials=n_trials,
        timeout=3600  # 1 hour timeout
    )
    
    print(f"\nOptimization finished for {animal_name}:")
    print(f"Best trial value: {study.best_trial.value:.2f}%")
    print("Best parameters:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    
    # Train final model with best parameters
    print(f"\nTraining final model for {animal_name} with best parameters...")
    model, history = train_final_model(study.best_trial, X, y, device)
    
    # Save the final model
    save_model(model, history, f"{animal_name}_optuna_final")
    print(f"Final model saved as {animal_name}_optuna_final")
    
    return study.best_trial

def main():
    # Define the classes
    tiger = ['tiger', 'Tiger_negative_class']
    elephant = ['elephant', 'Elephant_negative_class']
    fox = ['fox', 'Fox_negative_class']
    my_classes = [tiger, elephant, fox]
    
    # Dictionary to store best trials
    best_trials = {}
    
    for animal_class in my_classes:
        animal_name = animal_class[0]
        torch.cuda.empty_cache()
        print(f"\nOptimizing model for {animal_name}...")
        X, y = create_X_y(animal_class)
        
        # Run optimization and save final model
        best_trial = optimize_model(
            X, y, animal_name, n_trials=50
        )
        
        best_trials[animal_name] = best_trial
    
    return best_trials

if __name__ == '__main__':
    best_trials = main()