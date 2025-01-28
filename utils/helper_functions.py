import numpy as np
import random
import os
import cv2
import torch
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from config.config import Config

def create_training_data(list_classes):
    """
    Creates training data by reading images and assigning binary labels.
    Label 0 is assigned to any class containing 'negative', label 1 to others.
    
    Args:
        list_classes: List of class directory names (e.g., ['tiger', 'tiger_negative'])
    """
    training_data = []
    for classes in list_classes:
        class_num = 0 if 'negative' in classes.lower() else 1
        
        # Construct full path to class directory
        path = os.path.join(Config.PROCESSED_DATA_DIR, classes)
        
        for img in os.listdir(path):
            try:
                # Read and resize the image
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
                new_array = cv2.resize(img_array, (Config.IMG_SIZE, Config.IMG_SIZE))
                
                # Add the image and its label to our dataset
                training_data.append([new_array, class_num])
            except Exception as e:
                pass
    
    return training_data

def create_X_y (list_classes):
      training_data=create_training_data(list_classes)
      random.shuffle(training_data)
      X=[]
      y=[]
      for features, label in training_data:
        X.append(features)
        y.append(label)
      X=np.array(X).reshape(-1,Config.IMG_SIZE, Config.IMG_SIZE, 3)
      y=np.array(y)
      return X,y

def save_model(model, history, model_name, save_dir=Config.WEIGHTS_DIR):
    """
    Save the trained model and its training history.
    
    The model weights are saved as a .pth file, and the training history 
    (losses and accuracies) is saved as a .npz file for later analysis.
    """
    # Create directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Save model weights
    model_path = save_path / f"{model_name}_model.pth"
    torch.save(model.state_dict(), model_path)
    
    # Save training history (losses and accuracies)
    history_path = save_path / f"{model_name}_history.npz"
    np.savez(
        history_path,
        train_losses=history[0],
        val_losses=history[1],
        train_accuracies=history[2],
        val_accuracies=history[3]
    )
    print(f"Model and history saved for {model_name}")

def load_model(model_class, model_name, save_dir=Config.WEIGHTS_DIR, device=Config.device):
    """
    Load a previously saved model and its training history.
    Returns both the model and the training history for analysis.
    """
    save_path = Path(save_dir)
    
    # Initialize model and load weights
    model = model_class().to(device)
    model_path = save_path / f"{model_name}_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    
    # Load training history
    history_path = save_path / f"{model_name}_history.npz"
    history = dict(np.load(history_path))
    
    return model, history

def evaluate_model(model, test_loader, device, optimized=False):
    """
    Evaluate model performance on a test dataset.
    Returns predictions, true labels, and probabilities for analysis.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.inference_mode():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)            
            if optimized:
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                preds = (probs > 0.5).astype(float)
            else:
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
            
            labels = labels.squeeze().cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels)
    
    return np.array(all_preds), np.array(all_labels), np.array(all_probs)