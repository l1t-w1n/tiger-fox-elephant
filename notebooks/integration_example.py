# Ajoutez cette cellule à votre notebook pour tester les feature maps

# Fonction pour tester les feature maps avec votre modèle existant
def test_feature_maps_with_existing_model():
    """Test des feature maps avec le modèle BinaryCNN existant"""
    
    # Charger un modèle entraîné
    model = BinaryCNN().to(Config.device)
    
    # Charger les poids d'un modèle entraîné (remplacez par le bon chemin)
    try:
        model_path = Config.WEIGHTS_DIR / "elephant_simple_model.pth"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            print(f"Modèle chargé depuis {model_path}")
        else:
            print("Aucun modèle pré-entraîné trouvé. Utilisation du modèle non entraîné.")
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
    
    # Créer le visualisateur
    visualizer = FeatureMapVisualizer(model, device=Config.device)
    
    # Charger une image d'exemple depuis votre dataset
    data_dir = Config.DATA_DIR / "elephant" / "train"
    
    # Trouver une image d'exemple
    pos_dir = data_dir / "positive"
    neg_dir = data_dir / "negative"
    
    sample_images = []
    
    # Chercher des images dans le dossier positive
    if pos_dir.exists():
        for img_file in pos_dir.glob("*.jpg"):
            sample_images.append(("positive", img_file))
            if len(sample_images) >= 2:
                break
    
    # Chercher des images dans le dossier negative
    if neg_dir.exists() and len(sample_images) < 2:
        for img_file in neg_dir.glob("*.jpg"):
            sample_images.append(("negative", img_file))
            if len(sample_images) >= 2:
                break
    
    if not sample_images:
        print("Aucune image trouvée dans le dataset. Vérifiez le chemin des données.")
        return
    
    print(f"Images trouvées: {len(sample_images)}")
    
    # Test avec la première image
    label, image_path = sample_images[0]
    print(f"\n=== Analyse des Feature Maps pour une image {label} ===")
    print(f"Image: {image_path.name}")
    
    # Charger et prétraiter l'image
    try:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Impossible de charger l'image {image_path}")
            return
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Appliquer les mêmes transformations que votre dataset
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
        
        image_tensor = transform(image)
        
        # Visualiser l'image originale
        plt.figure(figsize=(6, 6))
        plt.imshow(image)
        plt.title(f'Image Originale ({label})')
        plt.axis('off')
        plt.show()
        
        # 1. Visualiser les feature maps de la première couche convolutionnelle
        print("\n1. Feature maps de la couche conv1:")
        visualizer.visualize_feature_maps(image_tensor, layer_name='conv1.0', num_features=16)
        
        # 2. Obtenir et afficher les statistiques
        feature_maps = visualizer.get_feature_maps(image_tensor)
        stats = analyze_feature_statistics(feature_maps)
        
        print("\n2. Statistiques des feature maps:")
        for layer_name, stat in stats.items():
            print(f"\n{layer_name}:")
            print(f"  Shape: {stat['shape']}")
            print(f"  Nombre de canaux: {stat['num_channels']}")
            print(f"  Valeur moyenne: {stat['mean']:.4f}")
            print(f"  Écart-type: {stat['std']:.4f}")
            print(f"  Min/Max: [{stat['min']:.4f}, {stat['max']:.4f}]")
        
        # 3. Test de prédiction
        print("\n3. Prédiction du modèle:")
        model.eval()
        with torch.no_grad():
            prediction = model(image_tensor.unsqueeze(0).to(Config.device))
            prob = torch.sigmoid(prediction).item()
            predicted_class = "positive" if prob > 0.5 else "negative"
            print(f"Probabilité: {prob:.4f}")
            print(f"Classe prédite: {predicted_class}")
            print(f"Classe réelle: {label}")
            print(f"Prédiction correcte: {'Oui' if predicted_class == label else 'Non'}")
        
        # 4. Si on a deux images, les comparer
        if len(sample_images) >= 2:
            print("\n4. Comparaison de deux images:")
            label2, image_path2 = sample_images[1]
            
            # Charger la deuxième image
            image2 = cv2.imread(str(image_path2), cv2.IMREAD_COLOR)
            if image2 is not None:
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image_tensor2 = transform(image2)
                
                print(f"Comparaison: {sample_images[0][0]} vs {sample_images[1][0]}")
                visualizer.compare_feature_maps(image_tensor, image_tensor2, 
                                               layer_name='conv1.0', num_features=8)
        
        # 5. Créer une heatmap d'activation simple
        print("\n5. Analyse d'activation:")
        try:
            # Version simplifiée de heatmap basée sur les gradients d'entrée
            model.eval()
            image_input = image_tensor.unsqueeze(0).to(Config.device)
            image_input.requires_grad_()
            
            output = model(image_input)
            
            # Calculer le gradient par rapport à l'entrée
            model.zero_grad()
            output.backward()
            
            gradients = image_input.grad.data[0]
            
            # Créer une heatmap simple
            heatmap = torch.mean(torch.abs(gradients), dim=0)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
            # Affichage
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Image originale
            axes[0].imshow(image)
            axes[0].set_title('Image Originale')
            axes[0].axis('off')
            
            # Heatmap
            axes[1].imshow(heatmap.cpu().numpy(), cmap='hot')
            axes[1].set_title('Heatmap d\'Activation')
            axes[1].axis('off')
            
            # Superposition
            axes[2].imshow(image, alpha=0.7)
            axes[2].imshow(heatmap.cpu().numpy(), cmap='hot', alpha=0.3)
            axes[2].set_title('Superposition')
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Erreur lors de la création de la heatmap: {e}")
        
    except Exception as e:
        print(f"Erreur lors du traitement de l'image: {e}")


# Fonction pour analyser l'évolution des feature maps pendant l'entraînement
def analyze_feature_evolution():
    """Analyse l'évolution des feature maps pendant l'entraînement"""
    print("=== Analyse de l'évolution des Feature Maps ===")
    
    # Modèle non entraîné
    model_untrained = BinaryCNN()
    
    # Modèle entraîné (si disponible)
    model_trained = BinaryCNN()
    model_path = Config.WEIGHTS_DIR / "elephant_simple_model.pth"
    
    if model_path.exists():
        model_trained.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("Comparaison: modèle non-entraîné vs modèle entraîné")
        
        # Prendre une image d'exemple
        data_dir = Config.DATA_DIR / "elephant" / "train" / "positive"
        if data_dir.exists():
            sample_image = None
            for img_file in data_dir.glob("*.jpg"):
                sample_image = img_file
                break
            
            if sample_image:
                # Charger l'image
                image = cv2.imread(str(sample_image))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
                ])
                
                image_tensor = transform(image)
                
                # Visualiser les feature maps des deux modèles
                viz_untrained = FeatureMapVisualizer(model_untrained)
                viz_trained = FeatureMapVisualizer(model_trained)
                
                # Obtenir les feature maps
                features_untrained = viz_untrained.get_feature_maps(image_tensor)['conv1.0'][0]
                features_trained = viz_trained.get_feature_maps(image_tensor)['conv1.0'][0]
                
                # Comparer quelques feature maps
                num_features = min(8, features_untrained.shape[0])
                
                fig, axes = plt.subplots(3, num_features, figsize=(20, 10))
                fig.suptitle('Évolution des Feature Maps: Non-entraîné vs Entraîné', fontsize=16)
                
                # Image originale (répétée)
                for i in range(num_features):
                    axes[0, i].imshow(image)
                    axes[0, i].set_title(f'Image Originale')
                    axes[0, i].axis('off')
                
                # Feature maps non-entraînées
                for i in range(num_features):
                    im = axes[1, i].imshow(features_untrained[i].cpu().numpy(), cmap='viridis')
                    axes[1, i].set_title(f'Non-entraîné - F{i}')
                    axes[1, i].axis('off')
                
                # Feature maps entraînées
                for i in range(num_features):
                    im = axes[2, i].imshow(features_trained[i].cpu().numpy(), cmap='viridis')
                    axes[2, i].set_title(f'Entraîné - F{i}')
                    axes[2, i].axis('off')
                
                plt.tight_layout()
                plt.show()
                
                # Statistiques comparatives
                stats_untrained = {
                    'mean': features_untrained.mean().item(),
                    'std': features_untrained.std().item(),
                    'activation_rate': (features_untrained > 0).float().mean().item()
                }
                
                stats_trained = {
                    'mean': features_trained.mean().item(),
                    'std': features_trained.std().item(),
                    'activation_rate': (features_trained > 0).float().mean().item()
                }
                
                print(f"\nStatistiques comparatives:")
                print(f"Non-entraîné - Moyenne: {stats_untrained['mean']:.4f}, "
                      f"Std: {stats_untrained['std']:.4f}, "
                      f"Taux d'activation: {stats_untrained['activation_rate']:.4f}")
                print(f"Entraîné - Moyenne: {stats_trained['mean']:.4f}, "
                      f"Std: {stats_trained['std']:.4f}, "
                      f"Taux d'activation: {stats_trained['activation_rate']:.4f}")
    else:
        print("Aucun modèle entraîné trouvé. Entraînez d'abord un modèle.")


# Lancer les tests
if __name__ == "__main__":
    print("=== Test des Feature Maps ===")
    test_feature_maps_with_existing_model()
    print("\n" + "="*50 + "\n")
    analyze_feature_evolution()
