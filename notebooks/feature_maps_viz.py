import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
import cv2
from pathlib import Path
import os
from datetime import datetime

class FeatureMapVisualizer:
    def __init__(self, model, device='cpu', output_base_dir="visualization_output"):
        self.model = model
        self.device = device
        self.model = self.model.to(device)  # Ensure model is on correct device
        self.model.eval()
        self.feature_maps = {}
        self.hooks = []
        self.output_base_dir = Path(output_base_dir)
        
    def _create_output_structure(self, model_name, image_name):
        """Creates organized folder structure for outputs"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.output_base_dir / f"{model_name}_{image_name}_{timestamp}"
        
        # Create subdirectories - keeping only all_layers and heatmaps
        dirs = {
            'all_layers': session_dir / "all_layers",
            'heatmaps': session_dir / "heatmaps"
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        return dirs
        
    def register_hooks(self):
        """Enregistre des hooks pour capturer les feature maps"""
        def hook_fn(name):
            def hook(module, input, output):
                self.feature_maps[name] = output.detach()
            return hook
        
        # Enregistrer des hooks pour les couches convolutionnelles
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Conv2d):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Supprime tous les hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_feature_maps(self, image_tensor):
        """Obtient les feature maps pour une image donnée"""
        self.feature_maps = {}
        self.register_hooks()
        
        with torch.no_grad():
            # Forward pass
            _ = self.model(image_tensor.unsqueeze(0).to(self.device))
        
        self.remove_hooks()
        return self.feature_maps
    
    def visualize_feature_maps(self, image_tensor, layer_name=None, num_features=16, 
                             save_path=None, show_plot=True):
        """Visualise les feature maps d'une couche spécifique"""
        feature_maps = self.get_feature_maps(image_tensor)
        
        if layer_name is None:
            # Prendre la première couche convolutionnelle
            layer_name = list(feature_maps.keys())[0]
        
        if layer_name not in feature_maps:
            print(f"Couche '{layer_name}' non trouvée. Couches disponibles: {list(feature_maps.keys())}")
            return
        
        features = feature_maps[layer_name][0]  # Premier échantillon du batch
        num_features = min(num_features, features.shape[0])
        
        # Calculer la grille
        cols = 4
        rows = (num_features + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
        fig.suptitle(f'Feature Maps - {layer_name}', fontsize=16)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_features):
            row = i // cols
            col = i % cols
            
            feature_map = features[i].cpu().numpy()
            
            im = axes[row, col].imshow(feature_map, cmap='viridis')
            axes[row, col].set_title(f'Feature {i}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col])
        
        # Masquer les axes non utilisés
        for i in range(num_features, rows * cols):
            row = i // cols
            col = i % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature maps saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def compare_feature_maps(self, image1, image2, layer_name=None, num_features=8, 
                           save_path=None, show_plot=True):
        """Compare les feature maps de deux images différentes"""
        # Get feature maps for both images
        features1 = self.get_feature_maps(image1)
        features2 = self.get_feature_maps(image2)
        
        if layer_name is None:
            layer_name = list(features1.keys())[0]
        
        if layer_name not in features1:
            print(f"Couche '{layer_name}' non trouvée.")
            return
        
        feat1 = features1[layer_name][0][:num_features].cpu().numpy()
        feat2 = features2[layer_name][0][:num_features].cpu().numpy()
        
        fig, axes = plt.subplots(2, num_features, figsize=(20, 6))
        fig.suptitle(f'Comparaison Feature Maps - {layer_name}', fontsize=16)
        
        for i in range(num_features):
            # Image 1
            im1 = axes[0, i].imshow(feat1[i], cmap='viridis')
            axes[0, i].set_title(f'Image 1 - Feature {i}')
            axes[0, i].axis('off')
            
            # Image 2
            im2 = axes[1, i].imshow(feat2[i], cmap='viridis')
            axes[1, i].set_title(f'Image 2 - Feature {i}')
            axes[1, i].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def visualize_all_layers(self, image_tensor, save_dir=None, show_plots=False):
        """Visualise les feature maps de toutes les couches"""
        feature_maps = self.get_feature_maps(image_tensor)
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for layer_name in feature_maps.keys():
            print(f"Visualisation de la couche: {layer_name}")
            save_path = save_dir / f"{layer_name.replace('.', '_')}_features.png" if save_dir else None
            self.visualize_feature_maps(image_tensor, layer_name, 
                                      save_path=save_path, show_plot=show_plots)


def load_and_preprocess_image(image_path, img_size=224):
    """Charge et prétraite une image pour la visualisation"""
    # Lecture de l'image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Transformation
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    
    return transform(image)


def analyze_feature_statistics(feature_maps, save_path=None):
    """Analyse statistique des feature maps"""
    stats = {}
    
    for layer_name, features in feature_maps.items():
        feat = features[0].cpu().numpy()  # Premier échantillon
        
        stats[layer_name] = {
            'shape': feat.shape,
            'mean': np.mean(feat),
            'std': np.std(feat),
            'min': np.min(feat),
            'max': np.max(feat),
            'num_channels': feat.shape[0] if len(feat.shape) >= 3 else 1
        }
    
    # Save statistics to file if path provided
    if save_path:
        with open(save_path, 'w') as f:
            f.write("=== Feature Maps Statistics ===\n\n")
            for layer_name, stat in stats.items():
                f.write(f"{layer_name}:\n")
                f.write(f"  Shape: {stat['shape']}\n")
                f.write(f"  Mean: {stat['mean']:.4f}\n")
                f.write(f"  Std: {stat['std']:.4f}\n")
                f.write(f"  Range: [{stat['min']:.4f}, {stat['max']:.4f}]\n")
                f.write(f"  Channels: {stat['num_channels']}\n\n")
        print(f"Statistics saved to: {save_path}")
    
    return stats


def create_activation_heatmap(model, image_tensor, target_class=None, device='cpu', 
                            save_path=None, original_image_path=None, show_plot=True):
    """Crée une heatmap d'activation pour visualiser les zones importantes"""
    model.eval()
    model = model.to(device)  # Ensure model is on correct device
    image_tensor = image_tensor.unsqueeze(0).to(device)
    image_tensor.requires_grad_()
    
    # Forward pass
    output = model(image_tensor)
    
    if target_class is None:
        target_class = output.argmax(dim=1)
    
    # Backward pass
    model.zero_grad()
    class_score = output[0, target_class]
    class_score.backward()
    
    # Obtenir les gradients
    gradients = image_tensor.grad.data[0]
    
    # Créer la heatmap
    heatmap = torch.mean(gradients, dim=0)
    heatmap = F.relu(heatmap)
    heatmap = heatmap / torch.max(heatmap)
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Image originale
    plt.subplot(1, 2, 1)
    if original_image_path:
        original_image = cv2.imread(str(original_image_path))
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        plt.imshow(original_image)
    else:
        # Use the tensor (denormalized)
        img_denorm = image_tensor[0].detach().cpu()
        # Denormalize
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_denorm = img_denorm * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        plt.imshow(img_denorm.permute(1, 2, 0))
    
    plt.title('Image Originale')
    plt.axis('off')
    
    # Heatmap
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap.cpu().numpy(), cmap='hot', interpolation='nearest')
    plt.title('Heatmap d\'Activation')
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return heatmap.cpu().numpy()


# Exemple d'utilisation avec sauvegarde organisée
def demo_feature_maps(model_path, image_path, config, save_all=True, show_plots=False):
    """Démonstration des feature maps avec sauvegarde - all layers et heatmaps seulement"""
    
    # Extract model and image names for folder organization
    model_name = Path(model_path).stem  # e.g., 'fox', 'tiger', 'elephant'
    image_name = Path(image_path).stem  # e.g., 'fox_130', 'tiger_220'
    
    # Charger le modèle
    from binary_cnn import ImprovedBinaryCNN
    
    model = ImprovedBinaryCNN()
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model = model.to(config.device)
    
    # Créer le visualisateur avec dossier de sortie organisé
    base_output_dir = "feature_visualization_results"
    visualizer = FeatureMapVisualizer(model, device=config.device, 
                                    output_base_dir=base_output_dir)
    
    # Create organized folder structure
    output_dirs = visualizer._create_output_structure(model_name, image_name)
    
    # Charger et prétraiter l'image
    image_tensor = load_and_preprocess_image(image_path, config.IMG_SIZE)
    
    print(f"=== Analyse des Feature Maps - {model_name.upper()} ===")
    print(f"Image: {image_name}")
    print(f"Output directory: {output_dirs['all_layers'].parent}")
    
    # 1. Visualiser toutes les couches
    print("\n1. Visualisation de toutes les couches:")
    all_layers_dir = output_dirs['all_layers'] if save_all else None
    visualizer.visualize_all_layers(image_tensor, save_dir=all_layers_dir, 
                                   show_plots=show_plots)
    
    # 2. Créer une heatmap d'activation
    print("\n2. Heatmap d'activation:")
    heatmap_path = output_dirs['heatmaps'] / "activation_heatmap.png" if save_all else None
    heatmap = create_activation_heatmap(model, image_tensor, save_path=heatmap_path,
                                      original_image_path=image_path, show_plot=show_plots,
                                      device=config.device)
    
    print(f"\n✅ Analyse terminée pour {model_name} - {image_name}")
    if save_all:
        print(f"📁 Tous les fichiers sauvegardés dans: {output_dirs['all_layers'].parent}")
    
    return output_dirs


def run_complete_analysis(config, save_all=True, show_plots=False):
    """Execute complete analysis for all models and save in organized folders"""
    
    project_root = Path.cwd()
    data_dir = project_root / "data" / "resized_and_split"
    weights_dir = project_root / "weights"
    
    # Model and image configurations
    models_config = [
        {
            'name': 'fox',
            'model_path': weights_dir / "fox.pth",
            'image_path': data_dir / "fox" / "test" / "positive" / "fox_1404.jpg"
        },
        {
            'name': 'tiger', 
            'model_path': weights_dir / "tiger.pth",
            'image_path': data_dir / "tiger" / "test" / "positive" / "tiger_260.jpg"
        },
        {
            'name': 'elephant',
            'model_path': weights_dir / "elephant.pth", 
            'image_path': data_dir / "elephant" / "test" / "positive" / "elephant_664.jpg"
        }
    ]
    
    all_output_dirs = []
    
    for model_config in models_config:
        print(f"\n{'='*60}")
        print(f"Processing {model_config['name'].upper()} model...")
        print(f"{'='*60}")
        
        try:
            output_dirs = demo_feature_maps(
                model_config['model_path'],
                model_config['image_path'], 
                config,
                save_all=save_all,
                show_plots=show_plots
            )
            all_output_dirs.append(output_dirs)
            
        except Exception as e:
            print(f"❌ Error processing {model_config['name']}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print("🎉 ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    
    if save_all and all_output_dirs:
        print(f"\n📁 All results saved in organized folders:")
        for i, dirs in enumerate(all_output_dirs):
            model_name = models_config[i]['name']
            print(f"  • {model_name.upper()}: {dirs['all_layers'].parent}")
    
    return all_output_dirs


if __name__ == "__main__":
    # Configuration
    class Config:
        IMG_SIZE = 224
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    config = Config()
    
    # Check if CUDA is available and working
    if config.device == 'cuda':
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    # Run complete analysis with organized saving
    print("Starting complete feature map analysis...")
    print(f"Using device: {config.device}")
    
    # Set save_all=True to save everything, show_plots=False to avoid showing plots
    run_complete_analysis(config, save_all=True, show_plots=False)