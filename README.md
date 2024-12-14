# Tiger-Fox-Elephant Classification Project

This project implements various deep learning approaches for animal image classification, focusing on distinguishing tigers, foxes, and elephants from other animals using computer vision techniques.

## Project Overview

Our project tackles the challenge of animal classification through a systematic exploration of different deep learning architectures and methodologies. Starting from a baseline CNN model, we progress through increasingly sophisticated approaches, including specialized architectures, data augmentation, transfer learning, and generative models.

## Features

We implement several key approaches to image classification:

1. **Baseline Classification**: A foundational CNN architecture establishing our performance baseline
2. **Specialized Models**: Custom architectures optimized for each animal category
3. **Data Augmentation**: Enhanced training through artificially expanded datasets
4. **Transfer Learning**: Leveraging pre-trained models (ResNet, VGG) for improved performance
5. **GAN Implementation**: Generating synthetic fox images and evaluating their quality
6. **Feature Visualization**: Understanding model decision-making through feature map analysis

## Project Structure

```
tiger-fox-elephant/
├── config/                # Configuration parameters and settings
├── data/                 # Dataset management
│   ├── raw/             # Original datasets
│   ├── processed/       # Preprocessed data
│   └── generated/       # GAN-generated images
├── models/              # Model implementations
│   ├── baseline/        # Base CNN architecture
│   ├── improved/        # Specialized models
│   ├── transfer_learning/
│   └── gan/            # GAN implementation
├── utils/               # Helper functions
├── notebooks/          # Experimental notebooks
├── checkpoints/        # Saved model states
├── results/            # Output and analysis
└── tests/              # Unit tests
```

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/tiger-fox-elephant.git
   cd tiger-fox-elephant
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The project uses three distinct datasets:

- Tiger vs. other animals (200 samples per class)
- Fox vs. other animals (200 samples per class)
- Elephant vs. other animals (200 samples per class)

Each dataset is balanced and preprocessed to ensure consistent image dimensions and formats.

## Model Architecture Overview

### Baseline Model

- Single CNN layer architecture
- Configurable hyperparameters
- Serves as a performance benchmark

### Specialized Models

- `modeleTiger`: Optimized for tiger classification
- `modeleFox`: Specialized for fox detection
- `modeleElephant`: Customized for elephant recognition

### Advanced Implementations

- Data Augmentation Models (IDG variants)
- Transfer Learning Models (TL variants)
- GAN for Fox Image Generation

## Usage

### Training Models

```python
from models.baseline import BaselineModel
from utils.data_preparation import prepare_data

# Load and prepare data
train_data, test_data = prepare_data()

# Initialize and train model
model = BaselineModel()
model.train(train_data, epochs=50)
```

### Evaluating Models

```python
# Evaluate on test data
results = model.evaluate(test_data)
```

### Generating Images (GAN)

```python
from models.gan import FoxGAN

gan = FoxGAN()
gan.generate_images(num_images=10)
```

## Results and Model Performance

Model performance metrics and comparisons will be documented in the `results/` directory, including:

- Classification accuracy
- Confusion matrices
- Learning curves
- Generated image samples
- Feature map visualizations

## Optional Extensions

1. **Image Colorization**: Implementation of autoencoders for grayscale to color conversion
2. **Classification Understanding**: Using LIME for model decision interpretation

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## References

## License

This project is opensource

## Authors

Maksym Lytvynenko
Léonard Rivals
Léo Quenette
Matis Bazireau

## Acknowledgments

- Project structure inspired by modern deep learning practices
- Dataset preparation guidelines from various computer vision projects
