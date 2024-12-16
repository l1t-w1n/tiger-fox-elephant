import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import sys
from pathlib import Path
project_root = Path.cwd().parent 
sys.path.append(str(project_root))
from config.config import Config
from utils.helper_functions import evaluate_model


def plot_curves_confusion (history,confusion_matrix,class_names):
  plt.figure(1,figsize=(16,6))
  plt.gcf().subplots_adjust(left = 0.125, bottom = 0.2, right = 1,
                          top = 0.9, wspace = 0.25, hspace = 0)

  # division de la fenêtre graphique en 1 ligne, 3 colonnes,
  # graphique en position 1 - loss fonction

  plt.subplot(1,3,1)
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['Training loss', 'Validation loss'], loc='upper left')
  # graphique en position 2 - accuracy
  plt.subplot(1,3,2)
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper left')

  # matrice de correlation
  plt.subplot(1,3,3)
  sns.heatmap(conf,annot=True,fmt="d",cmap='Blues',xticklabels=class_names, yticklabels=class_names)# label=class_names)
  # labels, title and ticks
  plt.xlabel('Predicted', fontsize=12)
  #plt.set_label_position('top')
  #plt.set_ticklabels(class_names, fontsize = 8)
  #plt.tick_top()
  plt.title("Correlation matrix")
  plt.ylabel('True', fontsize=12)
  #plt.set_ticklabels(class_names, fontsize = 8)
  plt.show()


def plot_curves(histories):
    plt.figure(1,figsize=(16,6))
    plt.gcf().subplots_adjust(left = 0.125, bottom = 0.2, right = 1,
                          top = 0.9, wspace = 0.25, hspace = 0)
    for i in range(len(histories)):
    	# plot loss
        plt.subplot(121)
        plt.title('Cross Entropy Loss')
        plt.plot(histories[i].history['loss'], color='blue', label='train')
        plt.plot(histories[i].history['val_loss'], color='red', label='test')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['Training loss', 'Validation loss'], loc='upper left')
        # plot accuracy
        plt.subplot(122)
        plt.title('Classification Accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.plot(histories[i].history['accuracy'], color='blue', label='train')
        plt.plot(histories[i].history['val_accuracy'], color='red',
                  label='test')
        plt.legend(['Training accuracy', 'Validation accuracy'], loc='upper left')
        plt.show()
        
def plot_examples(X,y):
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

def plot_model_performance(model, test_loader, animal_name, device):
    """
    Create comprehensive visualization of model performance including
    confusion matrix and prediction distribution.
    """
    predictions, true_labels, probabilities = evaluate_model(model, test_loader, device)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot confusion matrix
    confusion = np.zeros((2, 2))
    for pred, label in zip(predictions, true_labels):
        confusion[int(label), int(pred)] += 1
    
    im = ax1.imshow(confusion, interpolation='nearest', cmap='Blues')
    ax1.set_title(f'Confusion Matrix - {animal_name}')
    
    # Add text annotations to confusion matrix
    thresh = confusion.max() / 2
    for i in range(2):
        for j in range(2):
            ax1.text(j, i, int(confusion[i, j]),
                    ha="center", va="center",
                    color="white" if confusion[i, j] > thresh else "black")
    
    class_names = [f'Not {animal_name}', animal_name]
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(class_names)
    ax1.set_yticklabels(class_names)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    plt.colorbar(im, ax=ax1)
    
    # Plot probability distribution
    positive_probs = probabilities[true_labels == 1]
    negative_probs = probabilities[true_labels == 0]
    
    ax2.hist(negative_probs, bins=20, alpha=0.5, label=f'Not {animal_name}', color='red')
    ax2.hist(positive_probs, bins=20, alpha=0.5, label=animal_name, color='blue')
    ax2.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
    ax2.set_title('Prediction Probability Distribution')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Count')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()