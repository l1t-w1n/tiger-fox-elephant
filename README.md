# 🐯🦊🐘 HAI923I - Computer Vision & Deep Learning Project

> **Animal Image Classification, Generation and Super-Resolution**  
> University of Montpellier - Master 1 Computer Science (2024-2025)

## 📖 Overview

This project explores various deep learning approaches for **computer vision**, from classical image classification to state-of-the-art generative models. Through multiple experiments, we evaluate the effectiveness of different architectures and modern deep learning techniques applied to animal images (tigers, foxes, elephants).

### 🎯 Main Objectives

- **Classification**: Build CNNs for binary animal classification
- **Data Augmentation**: Experiment with Image Data Generation techniques
- **Transfer Learning**: Apply pre-trained ResNet-50 from ImageNet
- **Generative Models**: Explore VAE, GANs and diffusion models (DDPM)
- **Image Colorization**: Develop encoder-decoder and adversarial models
- **Super-Resolution**: Implement FECAN (Feature Enhanced Cascading Attention Network)
- **Multimodal Models**: Experiment with CLIP for text-guided generation

## 🚀 Key Results

### Classification Performance

- **CNN Baseline**: 78-83% accuracy
- **CNN + Data Augmentation**: 83-87% accuracy (+5%)
- **ResNet-50 Transfer Learning**: **98-99% accuracy** (+20%)

### Image Generation

- **VAE**: Colorization with partial structure preservation
- **GANs + U-Net**: Realistic and coherent colorization ✅
- **Diffusion Models**: Successful generation on cat faces ✅

## 📂 Project Structure

```
.
├── 📁 notebooks/                    # Jupyter Experiments
│   ├── binary_cnn.ipynb            # CNN baseline classification
│   ├── transfer_learning_ResNet50.ipynb
│   ├── koalarization_ae+ResNet_*.ipynb
│   └── UNet+cGan.ipynb
├── 📁 diffusion/                    # Diffusion models (DDPM)
│   ├── butterfly_tutorial.ipynb
│   ├── diffusion.final.py
│   └── simple_ddpm.ipynb
├── 📁 FECAN/                        # Super-resolution
│   ├── model.py                     # FECAN architecture (400+ lines)
│   ├── train.py
│   └── config.py
├── 📁 clip_generation_fail/         # CLIP + generation experiments
│   ├── clip+diffusion/
│   └── clip_2.0/
├── 📁 feature_visualization_results/ # Feature maps visualization
├── 📁 diffusion_samples/            # Diffusion-generated images
└── 📁 models/                       # Model architectures
```

## 🛠️ Installation & Requirements

### Environment Setup

```bash
# Clone the repository
git clone <repository_url>
cd tiger-fox-elephant-dl

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Main Dependencies

- **PyTorch** (vision + transformers)
- **Hugging Face** (diffusers, transformers)
- **OpenCV** (image processing)
- **Matplotlib, NumPy** (visualization)
- **TensorBoard** (logging)

## 🚀 Usage

### 1. Image Classification

```bash
# Train CNN baseline
jupyter notebook notebooks/binary_cnn.ipynb

# Transfer Learning with ResNet-50
jupyter notebook notebooks/transfer_learning_ResNet50.ipynb
```

### 2. Diffusion Models

```bash
# Image generation with DDPM
python diffusion/diffusion.final.py

# Diffusion tutorial (butterflies)
jupyter notebook diffusion/butterfly_tutorial.ipynb
```

### 3. Image Colorization

```bash
# Colorization with GANs + U-Net
jupyter notebook notebooks/UNet+cGan.ipynb

# Deep Koalarization approach
jupyter notebook notebooks/koalarization_ae+ResNet_fox.ipynb
```

### 4. Super-Resolution (FECAN)

```bash
cd FECAN/
python train.py
```

## 📊 Datasets Used

### Primary Datasets

- **Animals**: 57,253 images (19 classes)
  - Elephants: 12,037 images
  - Tigers: 6,976 images
  - Foxes: 6,499 images
- **Animal Faces**: 10,400 images
  - Cats: 5,400 centered faces
  - Dogs: 5,000 centered faces
- **Super-Resolution**: DIV2K + Flickr2K (3,650 HD images)

### Preprocessing

- Resizing: 224×224 (classification), 256×256 (generation)
- ImageNet normalization
- Data augmentation: rotations, flips, distortions, Gaussian noise
- Class balancing (1:1 ratio for binary classification)

## 🏆 Contributions & Innovations

### Developed Extensions

1. **Denoising Diffusion Probabilistic Models (DDPM)** - Animal image generation
2. **Advanced Colorization** - VAE, Deep Koalarization, GANs with U-Net
3. **FECAN Super-Resolution** - Complete reimplementation (400+ lines)
4. **CLIP Experiments** - Text-guided generation and style transfer

### Scientific Results

- **Feature maps visualization** of ResNet-50 for decision understanding
- **Comparative analysis** of diffusion architectures
- **Quantitative evaluation** of colorization techniques

## 📈 Detailed Performance

| Model          | Fox vs Others  | Tiger vs Others | Elephant vs Others |
| -------------- | -------------- | --------------- | ------------------ |
| CNN Baseline   | 78%            | 83%             | 82%                |
| CNN + Data Aug | 83% (+5%)      | 87% (+4%)       | 85% (+3%)          |
| ResNet-50 TL   | **98%** (+20%) | **99%** (+16%)  | **98%** (+16%)     |

## 🧪 Advanced Experiments

### Diffusion Models

- ✅ **Butterflies**: Successful generation (homogeneous dataset)
- ❌ **Foxes**: Failed (too much visual variability)
- ✅ **Cats**: High-quality generation (centered faces)

### CLIP + Generation

- ❌ **Direct guidance**: Blurry and non-discriminative images
- 🔄 **Style Transfer**: Under development (CLIPStyler)

## 👥 Team

- **Matis Bazireau** - Literature review, report writing
- **Maksym Lytvynenko** - Main technical development
- **Léo Quenette** - Data management, evaluation
- **Léonard Rivals** - Model architecture, validation

**Supervisor**: Pascal Poncelet (University of Montpellier)

## 🙏 Special Thanks

- **Mykola Nechay** - Technical guidance and contributions to generative models & CLIP experiments

## 📚 Scientific References

- **ResNet**: He et al., "Deep Residual Learning for Image Recognition"
- **DDPM**: Ho et al., "Denoising Diffusion Probabilistic Models"
- **CLIP**: Radford et al., "Learning Transferable Visual Models from Natural Language Supervision"
- **FECAN**: Huang et al., "Feature Enhanced Cascading Attention Network for Lightweight Image Super-Resolution" (Nature 2025)

## 🔮 Future Perspectives

### Planned Improvements

- [ ] **CLIPStyler** finalization for style transfer
- [ ] **Conditional diffusion models** optimization
- [ ] Complete **FECAN** benchmark vs other SR methods
- [ ] **CLIP fine-tuning** on specialized datasets

### Potential Applications

- Species classification for conservation
- Natural archive image restoration
- Educational scientific content generation

## 📄 Documentation

- **Complete technical report**: `TER_Rapport.pdf` (46 pages, French)
- **Project specifications**: `PROJET HAI923I - 2024-2025_TravailaFaire.pdf`
- **Training logs**: `logs/` directory
- **Visual results**: `*_samples/` and `feature_visualization_results/` directories

## 🏅 Academic Context

This project was developed as part of the **HAI923I** course in the Master 1 Computer Science program at the University of Montpellier, Faculty of Sciences, during the 2024-2025 academic year.

### Learning Outcomes

- Deep understanding of computer vision fundamentals
- Hands-on experience with state-of-the-art architectures
- Research methodology and experimental validation
- Technical writing and scientific communication

## 📜 License

This project is available under the **MIT License**.

## 🤝 Contributing

While this is primarily an academic project, we welcome:

- Bug reports and fixes
- Performance improvements
- Additional experiments and extensions
- Documentation enhancements

Please feel free to open issues or submit pull requests!

---

> 🎓 **Academic Project HAI923I - 2024/2025**  
> Master 1 Computer Science - University of Montpellier, Faculty of Sciences

_For technical questions or collaboration opportunities, please open an issue or contact the team._
