import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import simple_parsing
import numpy as np
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKL
from lumos_diffusion import DPMS_INTER
from utils.download import find_model
import lumos_diffusion.model.dino.vision_transformer as vits
import torchvision.transforms as T
from lumos_diffusion.model.lumos import LumosI2I_XL_2
from utils import find_model
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from transformers import CLIPModel, CLIPProcessor
from helper import create_combined_image_with_labels, freeze_model_parameters, create_transform

LEARN_EMB_PERTURBATION = True
src_txt = "a white dog"
trg_txt = "a black dog"

class DirectionLoss(torch.nn.Module):
    def __init__(self, loss_type='mse'):
        super(DirectionLoss, self).__init__()
        self.loss_type = loss_type
        self.loss_func = {
            'mse':    torch.nn.MSELoss,
            'cosine': torch.nn.CosineSimilarity,
            'mae':    torch.nn.L1Loss
        }[loss_type]()

    def forward(self, x, y):
        if self.loss_type == "cosine":
            return 1. - self.loss_func(x, y)
        
        return self.loss_func(x, y)

class CLIPLoss(torch.nn.Module):
    def __init__(self, clip_model, processor, loss_type='mse', lambda_direction=1.0):
        super(CLIPLoss, self).__init__()
        self.clip_model = clip_model
        self.processor = processor
        self.direction_loss = DirectionLoss(loss_type=loss_type)
        self.target_direction = None
        self.lambda_direction = lambda_direction

    def encode_images(self, img: torch.Tensor) -> torch.Tensor:
        """Encode images using CLIP model."""
        return self.clip_model.get_image_features(pixel_values=img)

    def get_text_features(self, text: str) -> torch.Tensor:
        """Get text features from CLIP model."""
        text_inputs = self.processor(text=[text], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(next(self.clip_model.parameters()).device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(
                input_ids=text_inputs['input_ids'], 
                attention_mask=text_inputs['attention_mask']
            )
        return text_features

    def prepare_image_for_clip(self, img: torch.Tensor, device=None) -> torch.Tensor:
        """Prepare raw image for CLIP (resize to 224x224 and normalize)."""
        if device is None:
            device = next(self.clip_model.parameters()).device

        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        img = img.to(device)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
        img = (img - mean) / std
        
        return img

    def get_image_features(self, img: torch.Tensor, norm: bool = True) -> torch.Tensor:
        image_features = self.encode_images(img)
        
        if norm:
            image_features /= image_features.clone().norm(dim=-1, keepdim=True)
        return image_features

    def compute_text_direction(self, source_class: str, target_class: str) -> torch.Tensor:
        source_features = self.get_text_features(source_class)
        target_features = self.get_text_features(target_class)
        text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
        text_direction /= text_direction.norm(dim=-1, keepdim=True)
        return text_direction

    def clip_directional_loss(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str) -> torch.Tensor:
        if self.target_direction is None:
            self.target_direction = self.compute_text_direction(source_class, target_class)

        src_encoding    = self.get_image_features(src_img)
        target_encoding = self.get_image_features(target_img)
        edit_direction = (target_encoding - src_encoding)
        edit_direction /= (edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7)

        return self.direction_loss(edit_direction, self.target_direction).mean()

    def forward(self, src_img: torch.Tensor, source_class: str, target_img: torch.Tensor, target_class: str, device=None) -> torch.Tensor:
        # Prepare images for CLIP if they're raw (not already normalized)
        src_img = self.prepare_image_for_clip(src_img, device)
        target_img = self.prepare_image_for_clip(target_img, device)
        clip_loss = self.lambda_direction * self.clip_directional_loss(src_img, source_class, target_img, target_class)
        return clip_loss

def generate_perturbed_image(
    embeddings,
    guidance_scale=4.5,
    seed=10,
    enable_grad=False
):
    vae, model = models["vae"], models["diffusion"]
    
    # Set random seed
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    
    # Ensure embeddings are on the correct device
    embeddings = embeddings.to(device)
    
    bsz = embeddings.shape[0]
    null_y = model.y_embedder.y_embedding[None].repeat(bsz, 1, 1)[:, None]
    z = torch.randn(1, 4, 32, 32, device=device).repeat(bsz, 1, 1, 1)
    model_kwargs = dict(mask=None)
    
    dpm_solver = DPMS_INTER(model.forward_with_dpmsolver,
                        condition=embeddings,
                        uncondition=null_y,
                        cfg_scale=guidance_scale,
                        model_kwargs=model_kwargs)
    output = dpm_solver.sample(
            z,
            steps=20,  # Default steps for multistep method
            order=2,
            skip_type="time_uniform",
            method="multistep",
            enable_grad=enable_grad)
    output = vae.decode(output / 0.18215).sample
    output = torch.clamp(output * 0.5 + 0.5, min=0, max=1)
    
    return output

def optimize_perturbed_image(
    prompt_img1,
    guidance_scale=4.5,
    seed=10,
    num_iterations=10,
    learning_rate=0.01,
    output_folder=None
):
    vae, dino, transform, model = models["vae"], models["vision_encoder"], models["transform"], models["diffusion"]
    
    # Ensure all models are frozen
    freeze_model_parameters(vae)
    freeze_model_parameters(dino)
    freeze_model_parameters(model)
    
    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    freeze_model_parameters(clip_model)
    
    prompt_img1_tensor = transform(prompt_img1).unsqueeze(0)
    
    with torch.no_grad():
        caption_emb = dino(prompt_img1_tensor.to(device))
        caption_emb = torch.nn.functional.normalize(caption_emb, dim=-1).unsqueeze(1).unsqueeze(1)
    
    variant_axes = nn.Parameter(torch.randn_like(caption_emb) * 0.01)
    
    optimizer = torch.optim.Adam([variant_axes], lr=learning_rate)
    
    loss_history = []
    iteration_images = []
    
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        iteration_folder = os.path.join(output_folder, "iterations")
        os.makedirs(iteration_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training variant_axes for {num_iterations} iterations")
    print(f"Source text: {src_txt}")
    print(f"Target text: {trg_txt}")
    print(f"Learning rate: {learning_rate}")
    print(f"{'='*60}\n")
    
    # Generate original image from original embedding for comparison
    with torch.no_grad():
        original_image = generate_perturbed_image(
            embeddings=caption_emb,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_grad=False
        )
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        perturbed_emb = caption_emb + variant_axes
        perturbed_emb = torch.nn.functional.normalize(perturbed_emb, dim=-1)
        
        generated_image = generate_perturbed_image(
            embeddings=perturbed_emb,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_grad=False
        )
        
        total_loss = compute_clip_loss(
            original_image=original_image,
            generated_image=generated_image,
            clip_model=clip_model,
            processor=processor,
            device=device
        )
        
        total_loss.backward()
        
        optimizer.step()
         
        loss_history.append(total_loss.item())
        
        with torch.no_grad():
            iter_image_copy = generated_image[0].clone().cpu()
            iter_image_np = (iter_image_copy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            iter_image_pil = Image.fromarray(iter_image_np)
            iteration_images.append(iter_image_pil)
            
            # Save to disk if output folder is provided
            if output_folder is not None:
                iter_image_path = os.path.join(iteration_folder, f"iteration_{iteration + 1:03d}.png")
                iter_image_pil.save(iter_image_path)
        
        if (iteration + 1) % max(1, num_iterations // 5) == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}:")
            print(f"  Total Loss: {total_loss.item():.4f}")
            print()
    
    with torch.no_grad():
        perturbed_emb = caption_emb + variant_axes
        generated_image = generate_perturbed_image(
            embeddings=perturbed_emb,
            guidance_scale=guidance_scale,
            seed=seed
        )
    
    return generated_image, variant_axes, loss_history, iteration_images

def compute_clip_loss(
    original_image,
    generated_image,
    clip_model,
    processor,
    device=None,
    lambda_direction=1.0
):
    x0 = original_image
    x = generated_image

    clip_loss_func = CLIPLoss(clip_model, processor, loss_type='cosine', lambda_direction=lambda_direction)
    clip_loss_value = clip_loss_func(x0, src_txt, x, trg_txt, device=device)
    
    loss_clip = (2 - clip_loss_value) / 2
    loss_clip = -torch.log(loss_clip)
    
    loss_l1 = nn.L1Loss()(x0, x)
    
    total_loss = 3 * loss_clip + 0.1 * loss_l1
    
    return total_loss


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser(description="Lumos Image Embedding Perturbation Demo")
    parser.add_argument("--vae-pretrained", type=str, default="stabilityai/sd-vae-ft-mse")
    parser.add_argument("--dino-type", type=str, default="vit_base")
    parser.add_argument("--dino-pretrained", type=str, default="./checkpoints/dino_vitbase16_pretrain.pth")
    parser.add_argument("--lumos-i2i-ckpt", type=str, default="./checkpoints/Lumos_I2I.pth")
    parser.add_argument("--port", type=int, default=19231)
    args = parser.parse_known_args()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # setting models
    models = dict()
    ## autoencoder
    weight_dtype = torch.float32
    vae = AutoencoderKL.from_pretrained(args.vae_pretrained).cuda()
    vae.eval()
    vae.to(weight_dtype)
    models["vae"] = vae
    ## vision encoder 
    dino = vits.__dict__[args.dino_type](patch_size=16, num_classes=0).cuda()
    state_dict = torch.load(args.dino_pretrained, map_location="cpu")
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = dino.load_state_dict(state_dict, strict=False)
    del state_dict
    dino.eval()
    models["vision_encoder"] = dino
    transform = T.Compose(create_transform(learn_img_perturbation=False))
    models["transform"] = transform
    ## diffusion model
    model_kwargs={"window_block_indexes": [], "window_size": 0, 
                    "use_rel_pos": False, "lewei_scale": 1.0, 
                    "caption_channels": dino.embed_dim, 'model_max_length': 1}
    # build models
    image_size = 256
    latent_size = int(image_size) // 8
    model = LumosI2I_XL_2(input_size=latent_size, **model_kwargs).to(device)
    state_dict = find_model(args.lumos_i2i_ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    model.to(weight_dtype)
    models["diffusion"] = model

    input_folder = os.path.join(os.path.dirname(__file__), "input")
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
    input_image_paths = sorted(
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if os.path.isfile(os.path.join(input_folder, f)) and os.path.splitext(f)[1].lower() in valid_exts
    )
    if not input_image_paths:
        raise FileNotFoundError(f"No valid images found in {input_folder}")

    output_root = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_root, exist_ok=True)
    aug_folder = os.path.join(output_root, "aug")
    os.makedirs(aug_folder, exist_ok=True)

    if LEARN_EMB_PERTURBATION:
        print(f"\n{'='*60}")
        print("LEARN_EMB_PERTURBATION mode: Optimizing variant_axes")
        print(f"{'='*60}")

        base_image_path = input_image_paths[0]
        prompt_img1 = Image.open(base_image_path)

        generated_image, variant_axes, loss_history, iteration_images = optimize_perturbed_image(
            prompt_img1=prompt_img1,
            guidance_scale=4.5,
            seed=10,
            num_iterations=10,
            learning_rate=0.01,
            output_folder=output_root
        )

        generated_image_np = (generated_image[0].cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(generated_image_np)
        output_path = os.path.join(aug_folder, "output_perturbed.png")

        create_combined_image_with_labels(prompt_img1, output_image, output_path)

        print(f"✓ Output saved to {output_path}")
        print(f"✓ Saved {len(iteration_images)} iteration images to {os.path.join(output_root, 'iterations')}")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"{'='*60}\n")
        prompt_img1.close()

