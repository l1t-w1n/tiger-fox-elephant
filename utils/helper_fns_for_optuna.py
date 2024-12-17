import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from config.config import Config
import matplotlib.pyplot as plt

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
        
        # Build fully connected layers using Sequential instead of ModuleList
        n_fc_layers = trial.suggest_int('n_fc_layers', 1, 2)
        fc_layers = []
        prev_size = flattened_size
        
        for i in range(n_fc_layers):
            fc_size = trial.suggest_int(f'n_fc_units_l{i}', 64, 512)
            fc_dropout = trial.suggest_float(f'fc_dropout_l{i}', 0.1, 0.5)
            
            fc_layers.extend([
                nn.Linear(prev_size, fc_size),
                nn.ReLU(),
                nn.Dropout(fc_dropout)
            ])
            prev_size = fc_size
        
        # Output layer without sigmoid (using BCEWithLogitsLoss)
        fc_layers.append(nn.Linear(prev_size, 1))
        
        # Wrap fc_layers in Sequential
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x):
        # Input validation
        if len(x.shape) != 4:
            raise ValueError(f"Expected 4D input tensor (batch_size, channels, height, width), got shape {x.shape}")
        
        # Forward pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Flatten and pass through fully connected layers
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

class MockTrial:
    """Mock trial class that mimics Optuna trial for model loading"""
    def __init__(self, params):
        self.params = params
        
    def suggest_int(self, name, low, high):
        return self.params[name]
        
    def suggest_float(self, name, low, high, log=False):
        return self.params[name]

def create_mock_trial(model_name):
    """Create a mock trial with the best parameters for each model"""
    if model_name == 'tiger_optuna_final':
        params = {
            'batch_size': 42,
            'lr': 2.5868972971292134e-05,
            'weight_decay': 5.183617813762396e-05,
            'n_layers': 2,
            'n_channels_l0': 108,
            'n_channels_l1': 22,
            'dropout_l0': 0.33437833350401747,
            'dropout_l1': 0.38473381150080915,
            'n_fc_layers': 1,
            'n_fc_units_l0': 434,
            'fc_dropout_l0': 0.2260000036956811,
            'n_epochs': 40
        }
    elif model_name == 'elephant_optuna_final':
        params = {
            'batch_size': 44,
            'lr': 4.707489846497084e-05,
            'weight_decay': 0.0003605984199756961,
            'n_layers': 2,
            'n_channels_l0': 94,
            'n_channels_l1': 88,
            'dropout_l0': 0.1706381816861283,
            'dropout_l1': 0.3029197744318907,
            'n_fc_layers': 1,
            'n_fc_units_l0': 210,
            'fc_dropout_l0': 0.1537288053113789,
            'n_epochs': 41
        }
    elif model_name == 'fox_optuna_final':
        params = {
            'batch_size': 39,
            'lr': 3.8204563468507654e-05,
            'weight_decay': 1.771607427297915e-05,
            'n_layers': 2,
            'n_channels_l0': 65,
            'n_channels_l1': 62,
            'dropout_l0': 0.32052513227864154,
            'dropout_l1': 0.15647822621024182,
            'n_fc_layers': 1,
            'n_fc_units_l0': 273,
            'fc_dropout_l0': 0.46858815052636377,
            'n_epochs': 25
        }
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return MockTrial(params)

def load_optimized_model(model_name, save_dir=Config.WEIGHTS_DIR, device=Config.device):
    """Load an optimized model using its best hyperparameters"""
    save_path = Path(save_dir)
    
    # Create mock trial with best parameters
    trial = create_mock_trial(model_name)
    
    # Initialize model with mock trial
    model = OptimizedCNN(trial).to(device)
    
    # Load weights
    model_path = save_path / f"{model_name}_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    # Load history
    history_path = save_path / f"{model_name}_history.npz"
    history = dict(np.load(history_path))
    
    return model, history