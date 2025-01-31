import torch
from pathlib import Path

class Config:
    # Project structure
    project_root = Path(__file__).parent.parent # This will point to the project root directory
    DATA_DIR = project_root / "data/resized_and_split/"
    WEIGHTS_DIR = project_root / "weights"
    
    # Image parameters
    IMG_SIZE = 224
    
    # Training hyperparameters
    BATCH_SIZE = 64
    LR = 1e-3
    EPOCHS = 20
    KFOLDS = 2
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers = 20
    
    # Random seed for reproducibility
    RANDOM_SEED = 42


if __name__ == '__main__':
    # Print paths for verification
    print(f"Project root: {Config.PROJECT_ROOT}")
    print(f"Data directory: {Config.DATA_DIR}")
    print(f"Processed data: {Config.PROCESSED_DATA_DIR}")
    
    # Print CUDA information
    if torch.cuda.is_available():
        print(f"\nCUDA Version: {torch.version.cuda}")
        print(f"Built with CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("\nCUDA is not available")