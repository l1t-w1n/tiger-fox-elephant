import torch

class Config:
    IMG_SIZE = 128
    COLUMNS = 25
    my_path = "./"
    RANDOM_SEED = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    print(Config.IMG_SIZE)
    if torch.cuda.is_available():
        
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"Built with CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
    else:
        print("CUDA is not available")
