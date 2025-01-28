import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import sys
from pathlib import Path
project_root = Path.cwd().parent 
sys.path.append(str(project_root))
from config.config import Config
from utils.helper_functions import evaluate_model

def plot_examples(X,y, columns=Config.COLUMNS):
    plt.figure(figsize=(15,15))
    for i in range(Config.COLUMNS):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        # cv2 lit met les images en BGR et matplotlib lit du RGB
        X[i] = cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB)
        plt.imshow(X[i]/255.,cmap=plt.cm.binary)
        plt.xlabel(str(y[i]))
def plot_all_examples(X, y):
    """
    Plots all images in the dataset using the same style as plot_examples.
    Automatically calculates the grid size based on the number of images.
    """
    # Calculate number of columns similar to original (5x5 grid)
    columns = 5
    
    # Calculate number of rows needed to show all images
    n_images = len(X)
    rows = (n_images + columns - 1) // columns  # Ceiling division
    
    # Create figure with appropriate size
    # Keep same size ratio as original (15x15 for 25 images)
    figure_size = 15 * (rows / 5)  # Scale figure height based on number of rows
    plt.figure(figsize=(15, figure_size))
    
    # Plot each image
    for i in range(n_images):
        plt.subplot(rows, columns, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # Convert color space from BGR to RGB, same as original
        img_rgb = cv2.cvtColor(X[i], cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb/255., cmap=plt.cm.binary)
        plt.xlabel(str(y[i]))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()

def plot_model_performance(model, test_loader, animal_name, device, optimized=False):
    """
    Creates a clear visualization of the confusion matrix with key performance metrics.
    """
    # Get model predictions and true labels
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device, optimized)
    
    # Calculate confusion matrix elements
    true_negatives = np.sum((predictions == 0) & (true_labels == 0))
    false_positives = np.sum((predictions == 1) & (true_labels == 0))
    false_negatives = np.sum((predictions == 0) & (true_labels == 1))
    true_positives = np.sum((predictions == 1) & (true_labels == 1))
    
    # Create confusion matrix
    confusion = np.array([[true_negatives, false_positives],
                         [false_negatives, true_positives]])
    
    # Calculate performance metrics
    total_samples = len(true_labels)
    accuracy = (true_positives + true_negatives) / total_samples
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Create figure for confusion matrix
    plt.figure(figsize=(8, 6))
    im = plt.imshow(confusion, interpolation='nearest', cmap='Blues')
    plt.title(f'Confusion Matrix - {animal_name}\nAccuracy: {accuracy:.1%}')
    
    # Add text annotations to matrix
    thresh = confusion.max() / 2
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f'{int(confusion[i, j])}\n({confusion[i, j]/total_samples:.1%})',
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    
    # Set axis labels
    class_names = [f'Not {animal_name}', animal_name]
    plt.xticks([0, 1], class_names)
    plt.yticks([0, 1], class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # Add colorbar
    plt.colorbar(im)
    
    # Add metrics text box
    metrics_text = (f'Total samples: {total_samples}\n'
                   f'Precision: {precision:.1%}\n'
                   f'Recall: {recall:.1%}')
    plt.text(1.5, 0.5, metrics_text, transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    # Print verification data
    print(f"\nModel Predictions Shape: {predictions.shape}")
    print(f"True Labels Shape: {true_labels.shape}")
    print("\nDetailed Metrics:")
    print(f"True Negatives: {true_negatives} ({true_negatives/total_samples:.1%})")
    print(f"False Positives: {false_positives} ({false_positives/total_samples:.1%})")
    print(f"False Negatives: {false_negatives} ({false_negatives/total_samples:.1%})")
    print(f"True Positives: {true_positives} ({true_positives/total_samples:.1%})")
 
def visualize_misclassified(model, test_loader, animal_name, device, num_images=25, optimized=False):
    """
    Visualizes images that the model misclassified
    """
    model.eval()
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []
    misclassified_probs = []
    
    with torch.inference_mode():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            if optimized:
                probabilities = torch.sigmoid(outputs).squeeze().cpu().numpy()
            else:
                probabilities = outputs.cpu().numpy()
                
            labels = labels.cpu().numpy()            
            predictions = (probabilities > 0.5).astype(float)
                        
            """ print("Predictions: ", predictions)
            print("Probabilities: ", probabilities)
            print("Labels: ", labels) """
            
            # Find misclassified images in this batch
            incorrect_mask = predictions != labels
            
            if incorrect_mask.any():
                batch_incorrect = images[incorrect_mask]
                batch_probs = probabilities[incorrect_mask]
                batch_preds = predictions[incorrect_mask]
                batch_labels = labels[incorrect_mask]
                
                # Add to our collection
                misclassified_images.extend(batch_incorrect)
                misclassified_preds.extend(batch_preds)
                misclassified_labels.extend(batch_labels)
                misclassified_probs.extend(batch_probs)
            
            if len(misclassified_images) >= num_images:
                break
    
    # Limit to requested number of images
    misclassified_images = misclassified_images[:num_images]
    misclassified_preds = misclassified_preds[:num_images]
    misclassified_labels = misclassified_labels[:num_images]
    misclassified_probs = misclassified_probs[:num_images]
    
    # Calculate grid dimensions
    n_images = len(misclassified_images)
    if n_images == 0:
        print("No misclassified images found!")
        return
    
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    
    plt.figure(figsize=(15, 3 * rows))
    
    for idx in range(n_images):
        # Convert tensor to numpy array in BGR format
        # We need to permute to (H,W,C) format first
        img = misclassified_images[idx].cpu().permute(1, 2, 0).numpy()
        
        # Scale back to 0-255 range for cv2.cvtColor
        img = (img * 255).astype(np.uint8)
        
        # Convert from BGR to RGB using cv2.cvtColor
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize back to 0-1 range for plotting
        img = img / 255.0
        
        pred = bool(misclassified_preds[idx])
        true_label = bool(misclassified_labels[idx])
        prob = misclassified_probs[idx]
        
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.axis('off')
        
        title = f'True: {"" if true_label else "Not "}{animal_name}\n'
        title += f'Pred: {"" if pred else "Not "}{animal_name}\n'
        title += f'Conf: {prob:.2%}'
        
        plt.title(title, color='red', fontsize=10)
    
    plt.suptitle(f'Misclassified {animal_name} Images', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics about errors
    false_positives = sum((np.array(misclassified_preds) == 1) & (np.array(misclassified_labels) == 0))
    false_negatives = sum((np.array(misclassified_preds) == 0) & (np.array(misclassified_labels) == 1))
    
    print("\nError Analysis Summary:")
    print(f"Total misclassified images shown: {n_images}")
    print(f"False Positives (incorrectly predicted as {animal_name}): {false_positives}")
    print(f"False Negatives (incorrectly predicted as not {animal_name}): {false_negatives}")