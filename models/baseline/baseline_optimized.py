import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from config.config import Config
from utils.helper_functions import create_X_y

class AnimalDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).permute(0, 3, 1, 2) / 255.0
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class BinaryCNN(nn.Module):
    def __init__(self, trial):
        """
        Modified CNN architecture with Optuna trial parameters
        """
        super(BinaryCNN, self).__init__()
        
        # Optimize number of convolutional layers
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        
        # First layer always starts with 3 input channels (RGB)
        in_channels = 3
        layers = []
        
        # Build convolutional layers dynamically
        for i in range(n_conv_layers):
            # Optimize number of output channels
            out_channels = trial.suggest_int(f'conv{i}_out_channels', 16, 64)
            
            # Optimize kernel size
            kernel_size = trial.suggest_int(f'conv{i}_kernel', 3, 5)
            
            # Add convolutional block
            layers.append(nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size//2
            ))
            layers.append(nn.ReLU())
            
            # Optimize pooling size
            pool_size = trial.suggest_int(f'pool{i}_size', 2, 3)
            layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate size of flattened features after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Config.IMG_SIZE[0], Config.IMG_SIZE[0])
            dummy_output = self.conv_layers(dummy_input)
            n_features = dummy_output.numel() // dummy_output.shape[0]
        
        # Build fully connected layers
        fc_layers = []
        
        # Optimize number of fully connected layers
        n_fc_layers = trial.suggest_int('n_fc_layers', 1, 3)
        
        # Optimize first FC layer size
        fc_in = n_features
        
        for i in range(n_fc_layers):
            fc_out = trial.suggest_int(f'fc{i}_units', 32, 256)
            fc_layers.extend([
                nn.Linear(fc_in, fc_out),
                nn.ReLU(),
                nn.Dropout(trial.suggest_float(f'dropout{i}', 0.2, 0.5))
            ])
            fc_in = fc_out
        
        # Final classification layer
        fc_layers.extend([
            nn.Linear(fc_in, 1),
            nn.Sigmoid()
        ])
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x.squeeze()

def objective(trial, X, y, device):
    """
    Optuna objective function for optimizing the model
    """
    # Create data splits
    train_size = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    X_train, y_train = X[train_indices], y[train_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    
    # Create datasets
    train_dataset = AnimalDataset(X_train, y_train)
    val_dataset = AnimalDataset(X_val, y_val)
    
    # Optimize batch size
    batch_size = trial.suggest_int('batch_size', 16, 64)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model with trial parameters
    model = BinaryCNN(trial).to(device)
    
    # Optimize learning rate
    lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    
    # Optimize optimizer choice
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    criterion = nn.BCELoss()
    
    # Training loop
    n_epochs = 20  # Reduced epochs for optimization
    best_val_acc = 0
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Validation phase
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                predicted = (outputs > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_acc = 100 * correct / total
        
        # Report intermediate value
        trial.report(val_acc, epoch)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
        
        best_val_acc = max(best_val_acc, val_acc)
    
    return best_val_acc

def optimize_hyperparameters(X, y, n_trials=100):
    """
    Run the hyperparameter optimization
    """
    device = torch.device(Config.device)
    
    # Create study object
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=5
        )
    )
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, X, y, device),
        n_trials=n_trials,
        timeout=3600  # 1 hour timeout
    )
    
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    
    return study

def main():
    # Load your data
    my_path = Config.PROCESSED_DATA_DIR
    animal_classes = [
        ['tiger', 'Tiger_negative_class'],
        ['elephant', 'Elephant_negative_class'],
        ['fox', 'Fox_negative_class']
    ]
    
    for animal_class in animal_classes:
        print(f"\nOptimizing model for {animal_class[0]}...")
        X, y = create_X_y(my_path, animal_class)
        
        # Run optimization
        study = optimize_hyperparameters(X, y)
        
        # Save optimization results
        results_path = Config.RESULTS_DIR / f"{animal_class[0]}_optuna_results.pkl"
        optuna.save_study(study, results_path)

if __name__ == "__main__":
    main()