import torch
from pathlib import Path

class Config:
    # Project structure
    PROJECT_ROOT = Path(__file__).parent.parent # This will point to the project root directory
    DATA_DIR = PROJECT_ROOT / "data"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    RAW_DATA_DIR = DATA_DIR / "raw"
    MODELS_DIR = PROJECT_ROOT / "models"
    RESULTS_DIR = PROJECT_ROOT / "results"
    WEIGHTS_DIR = MODELS_DIR / "weights"
    
    # Model parameters
    IMG_SIZE = 128
    COLUMNS = 25
    RANDOM_SEED = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

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