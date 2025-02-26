import torch

WIDTH=512
HEIGHT=512
CHANNELS=3
assert WIDTH % 8 == 0
assert HEIGHT % 8 == 0
LATENT_WIDTH=WIDTH // 8
LATENT_HEIGHT=HEIGHT // 8
 
BATCH_SIZE=4
NUM_WORKERS= 16
NUM_EPOCHS=100
LEARNING_RATE=1e-4
NUM_TIME_STEPS=1000
NUM_INFERENCE_STEPS=200

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42

checkpoint_path = '~/tiger-fox-elephant/ddpm/src/data/v1-5-pruned-emaonly.ckpt'
