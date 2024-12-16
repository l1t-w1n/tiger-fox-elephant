# Tiger-Fox-Elephant Classification Project

This project implements various deep learning approaches for animal image classification, focusing on distinguishing tigers, foxes, and elephants from other animals using computer vision techniques.

## Project Overview

Our project tackles the challenge of animal classification through a systematic exploration of different deep learning architectures and methodologies. Starting from a baseline CNN model, we progress through increasingly sophisticated approaches, including specialized architectures, data augmentation, transfer learning, and generative models.

## Features

We implement several key approaches to image classification:

1. **Baseline Classification**: A foundational CNN architecture establishing our performance baseline
2. **Data Augmentation**: Enhanced training through artificially expanded datasets
3. **Transfer Learning**: Leveraging pre-trained models (ResNet, VGG) for improved performance
4. **GAN Implementation**: Generating synthetic fox images and evaluating their quality
5. **Feature Visualization**: Understanding model decision-making through feature map analysis

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
   git clone https://github.com/l1t-w1n/tiger-fox-elephant.git
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

- Maksym Lytvynenko
- Léonard Rivals
- Léo Quenette
- Matis Bazireau

## Acknowledgments

- Project structure inspired by modern deep learning practices
- Dataset preparation guidelines from various computer vision projects
