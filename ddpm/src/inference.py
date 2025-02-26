import torch 
import config
import numpy as np
from PIL import Image
from tqdm import tqdm 
from ddpm import DDPMSampler
from transformers import CLIPTokenizer

import model_loader
import config

class InferencePipeline:
    def __init__(
        self, 
        prompt: str,
        unconditional_prompt: str,
        image: torch.Tensor,
        strength: float,
        do_cfg: bool,
        cfg_scale: float = 7.5,
        sampler_name: str = 'ddpm',
        steps: int = 50,
        device: str = 'cpu',
        seed: int = 42,
        models: dict = {},
        tokenizer = None
    ):
        
        self.prompt = prompt
        self.unconditional_prompt = unconditional_prompt
        self.image = image
        self.strength = strength
        self.do_cfg = do_cfg
        self.cfg_scale = cfg_scale
        self.sampler_name = sampler_name
        self.steps = steps
        self.device = device
        self.seed = seed
        self.models = models
        self.tokenizer = tokenizer
        
    def generate(self):
        with torch.no_grad():
            if not (self.cfg_scale > 0 and self.cfg_scale < 14):
                raise  ValueError('cfg_scale must be between 0 and 14')
            elif self.strength < 0 or self.strength > 1:
                raise ValueError('strength must be between 0 and 1')
            
            generator = torch.Generator(device=self.device)
            if self.seed is None:
                generator.manual_seed(torch.seed())
            else:
                generator.manual_seed(self.seed)            
            
            clip = self.models['clip']
            clip = clip.to(self.device)
            clip.eval()
            
            context = None
            
            if self.do_cfg:
                # Convert the prompt into tokens 
                prompt_tokens = self.tokenizer.batch_encode_plus(
                    [
                        self.prompt
                    ],
                    padding='max_length',
                    max_length=77
                ).input_ids
                
                prompt_token_tensors = torch.tensor(prompt_tokens, dtype=torch.long).to(self.device)
                
                # Convert (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
                prompt_token_tensors = clip(prompt_token_tensors)
                
                unconditional_prompt_tokens = self.tokenizer.batch_encode_plus([self.unconditional_prompt], padding='max_length', max_length=77).input_ids
                unconditional_prompt_token_tensors = torch.tensor(unconditional_prompt_tokens, dtype=torch.long).to(self.device)
                
                # Convert (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
                unconditional_prompt_token_tensors = clip(unconditional_prompt_token_tensors)
                
                # [2, 77, 768]
                context = torch.cat([prompt_token_tensors, unconditional_prompt_token_tensors], dim=1)
            else:
                # Convert the prompt into tokens 
                prompt_tokens = self.tokenizer.batch_encode_plus([self.prompt], padding='max_length', max_length=77).input_ids
                prompt_token_tensors = torch.tensor(prompt_tokens, dtype=torch.long).to(self.device)
                
                # Convert (batch_size, sequence_length) -> (batch_size, sequence_length, embedding_dim)
                context = clip(prompt_token_tensors)
                

            if self.sampler_name == 'ddpm':
                sampler = DDPMSampler(
                    generator
                )
                sampler.set_inference_timesteps(self.steps)
            else:
                raise ValueError('Unknown sampler name')
            
            
            latents_shape = (1, 4, config.LATENT_HEIGHT, config.LATENT_WIDTH)
            
            # If image is provided, encode the image and get the latents
            if self.image:
                encoder = self.models['encoder']
                encoder = encoder.to(self.device)
                encoder.eval()
                
                # Load and preprocess the image
                image = Image.open(self.image).convert('RGB')  # Convert to RGB to handle alpha channels
                image = image.resize((config.HEIGHT, config.WIDTH))  # Resize using PIL
                image_array = np.array(image)  # Now convert to numpy array

                # Create tensor from the properly resized image
                image_tensor = torch.tensor(image_array, dtype=torch.float32, device=self.device)
                
                
                rescaled_image = self._rescale_image(image_tensor, (0, 255), (-1, 1))
                rescaled_image = rescaled_image.unsqueeze(0).permute(0, 3, 1, 2)
                
                # Now we add specified noise level to the latents to add some variance in the generation process
                gaussian_noise = torch.randn(latents_shape, generator=generator, device=self.device)
                
                # (Batch_Size, 4, Latents_Height, Latents_Width)
                latents = encoder(
                    rescaled_image,
                    gaussian_noise
                )
                
                sampler.set_strength(strength=self.strength)
                latents, _ = sampler.add_noise(latents, sampler.timesteps[0])
                
            else:
                # If we don`t specift the input image, we will state from pure noise Gaussian N(0, I)
                latents = torch.randn(latents_shape, generator=generator, device=self.device)
                
            diffusion_model = self.models['diffusion']
            diffusion_model = diffusion_model.to(self.device)
            diffusion_model.eval()
            
            timesteps = tqdm(sampler.timesteps, desc='Generating images')
            for i, timestep in enumerate(timesteps):
                
                # Now we pass through the diffusion model pipeline and for every denoisification step we will also indicate the timestep
                # scalar value -> (1, 320)
                time_embedding = self.get_time_embedding(timestep)
                model_input = latents
                
                if self.do_cfg:
                    # (batch_size, 4, latent_height, latent_width) -> (2 * batch_size, 4, latent_height, latent_width)
                    model_input = model_input.repeat(2, 1, 1, 1)
                    
                    
                model_output = diffusion_model(
                    latent=model_input,
                    context=context,
                    time=time_embedding
                )
                
                if self.do_cfg:
                    conditional_output, unconditional_output = torch.chunk(model_output, 2, dim=0)
                    model_output = self.cfg_scale * (conditional_output - unconditional_output) + unconditional_output
                    
                # This step is delete defined noise level == model_output from the latents on each step (timestep) of the diffusification process
                # step - this method directly delete the noise level from the latents
                latents = sampler.step(timestep, latents, model_output)
                
            # After the diffusion loop (denoising steps)
            latents = latents / 0.18215  # Scale latents for the decoder

            decoder = self.models['decoder']
            decoder = decoder.to(self.device)
            decoder.eval()

            images = decoder(latents)

            # Rescale, convert to uint8, and adjust dimensions
            images = self._rescale_image(images, (-1, 1), (0, 255), clamp=True)
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            images = images.astype(np.uint8)  # Ensure dtype is uint8

        return images[0]  # Shape: (H, W, 3)
        
    def _rescale_image(self, image: torch.Tensor, input_range: tuple, output_range: tuple, clamp: bool = False):
        """
            Rescale the image tensor from input_range to output_range
        """
        input_min, input_max = input_range
        output_min, output_max = output_range
        
        image = (image - input_min) * ((output_max - output_min) / (input_max - input_min)) + output_min
                
        if clamp:
            image = torch.clamp(image, min=output_min, max=output_max)
        return image
        
    def get_time_embedding(self, timestep):
        # Shape: (160,)
        freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32, device=self.device) / 160) 
        # Shape: (1, 160)
        x = torch.tensor([timestep], dtype=torch.float32, device=self.device)[:, None] * freqs[None]
        # Shape: (1, 160 * 2)
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)
            
            
            
if __name__ == '__main__':
    tokenizer = CLIPTokenizer('/home/nikolay/thesis/tiger-fox-elephant/ddpm/src/data/tokenizer_vocab.json', '/home/nikolay/thesis/tiger-fox-elephant/ddpm/src/data/tokenizer_merges.txt')
    models = model_loader.preload_models_from_standard_weights('/home/nikolay/thesis/tiger-fox-elephant/ddpm/src/data/v1-5-pruned-emaonly.ckpt', config.DEVICE)
    
    prompt = "A butterfly in the forest, highly detailed, ultra sharp, cinematic, 100mm lens, 8k resolution."
    unconditional_prompt = ""
    do_cfg = True
    cfg_scale = 8.0
    
    input_image = Image.open('/home/nikolay/thesis/tiger-fox-elephant/ddpm/src/data/butterflies_dataset/images/?irn=14025922.jpg')
    image_path = '/home/nikolay/thesis/tiger-fox-elephant/ddpm/src/data/butterflies_dataset/images/?irn=14025922.jpg'
    strength = 0.7
    sampler = 'ddpm'
    
    inference = InferencePipeline(
        prompt=prompt,
        unconditional_prompt=unconditional_prompt,
        image=image_path,
        strength=strength,
        do_cfg=do_cfg,
        cfg_scale=cfg_scale,
        sampler_name=sampler,
        steps=config.NUM_INFERENCE_STEPS,
        seed=config.SEED,
        models=models,
        device=config.DEVICE,
        tokenizer=tokenizer,
    )
    output_image = inference.generate()
    # Combine the input image and the output image into a single image.
    Image.fromarray(output_image)
    
    Image.fromarray(output_image).save('/home/nikolay/thesis/tiger-fox-elephant/ddpm/src/data/output.jpg')