#!/bin/bash

# Store the project root directory name
PROJECT_ROOT="tiger-fox-elephant"

# Create the main project directory if it doesn't exist
if [ ! -d "$PROJECT_ROOT" ]; then
    mkdir "$PROJECT_ROOT"
fi

# Navigate into the project directory
cd "$PROJECT_ROOT"

# Create the directory structure with a function to make it more organized
create_directories() {
    # Config directory
    mkdir -p config
    touch config/__init__.py
    touch config/config.py

    # Data directory with its subdirectories
    mkdir -p data/{raw/{tiger,fox,elephant},processed,generated}

    # Models directory with all its subdirectories
    mkdir -p models/{baseline,improved,transfer_learning,gan}
    touch models/__init__.py
    touch models/baseline/__init__.py
    touch models/baseline/baseline_cnn.py
    touch models/improved/__init__.py
    touch models/improved/{tiger_model,fox_model,elephant_model}.py
    touch models/transfer_learning/__init__.py
    touch models/transfer_learning/pretrained_models.py
    touch models/gan/__init__.py
    touch models/gan/fox_gan.py

    # Utils directory
    mkdir -p utils
    touch utils/__init__.py
    touch utils/{data_preparation,visualization,metrics}.py

    # Notebooks directory
    mkdir -p notebooks
    touch notebooks/{1_baseline_experiments,2_improved_models,3_data_augmentation,4_transfer_learning,5_gan_generation}.ipynb

    # Other important directories
    mkdir -p {checkpoints,results/{figures,metrics,generated_images},tests}
    touch tests/__init__.py

    # Root level files
    touch {requirements.txt,setup.py,README.md}
}

# Create all directories and files
create_directories

# Initial content for README.md
cat > README.md << 'EOL'
# Tiger-Fox-Elephant Classification Project

## Project Structure
This project is organized as follows:

- `config/`: Configuration files and parameters
- `data/`: Dataset storage
  - `raw/`: Original datasets
  - `processed/`: Preprocessed datasets
  - `generated/`: Generated images
- `models/`: Model implementations
- `utils/`: Helper functions and utilities
- `notebooks/`: Jupyter notebooks for experiments
- `checkpoints/`: Saved model states
- `results/`: Output and analysis
- `tests/`: Unit tests

## Setup
[Instructions for setting up the project will go here]

## Usage
[Usage instructions will go here]
EOL

# Initial content for requirements.txt
cat > requirements.txt << 'EOL'
numpy
pandas
torch
torchvision
matplotlib
seaborn
opencv-python
tqdm
jupyter
EOL

# Print success message
echo "Project structure created successfully!"
echo "You can now cd into $PROJECT_ROOT and begin development."
echo "Don't forget to:"
echo "1. Initialize a git repository"
echo "2. Create a virtual environment"
echo "3. Install requirements using 'pip install -r requirements.txt'"
