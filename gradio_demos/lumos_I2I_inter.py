import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import simple_parsing
import numpy as np
import torch
from torchvision.utils import make_grid
import random
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

INTERPOLATION = False
INTERPRETATION = True
INFO_GRAD = False

MAX_SEED = 2147483647
def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def dividable(n):
    for i in range(int(np.sqrt(n)), 0, -1):
        if n % i == 0:
            break
    return i, n // i

def print_frozen_modules(model, model_name):
    """
    Print which modules have requires_grad=False (frozen parameters).
    
    Args:
        model: PyTorch model
        model_name: Name of the model for printing
    """
    print(f"\n{'='*60}")
    print(f"Checking frozen parameters for: {model_name}")
    print(f"{'='*60}")
    
    frozen_modules = []
    trainable_modules = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            frozen_modules.append(name)
        else:
            trainable_modules.append(name)
    
    if frozen_modules:
        print(f"\nFrozen modules (requires_grad=False): {len(frozen_modules)}")
        print("-" * 60)
        for name in frozen_modules[:20]:  # Print first 20
            print(f"  {name}")
        if len(frozen_modules) > 20:
            print(f"  ... and {len(frozen_modules) - 20} more")
    else:
        print("\nNo frozen modules found (all parameters have requires_grad=True)")
    
    if trainable_modules:
        print(f"\nTrainable modules (requires_grad=True): {len(trainable_modules)}")
        print("-" * 60)
        for name in trainable_modules[:20]:  # Print first 20
            print(f"  {name}")
        if len(trainable_modules) > 20:
            print(f"  ... and {len(trainable_modules) - 20} more")
    else:
        print("\nNo trainable modules found (all parameters have requires_grad=False)")
    
    total_params = len(frozen_modules) + len(trainable_modules)
    print(f"\nSummary: {len(frozen_modules)}/{total_params} frozen, {len(trainable_modules)}/{total_params} trainable")

def create_combined_image_with_labels(input_image, output_image, output_path):
    """
    Create a combined image with input and output images side by side with labels using matplotlib.
    
    Args:
        input_image: PIL Image - the input image
        output_image: PIL Image - the output image
        output_path: str - path to save the combined image
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Image Interpretation', fontsize=16, fontweight='bold')
    
    # Display input image
    axes[0].imshow(input_image)
    axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    # Display output image
    axes[1].imshow(output_image)
    axes[1].set_title('Output Image', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def generate(
    prompt_img1,
    prompt_img2=None,
    bsz=1,
    guidance_scale=4.5,
    num_inference_steps=20,
    seed=10,
    randomize_seed=True,
    method="multistep"
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    vae, dino, transform, model = models["vae"], models["vision_encoder"], models["transform"], models["diffusion"]
    prompt_img1 = transform(prompt_img1).unsqueeze(0)
    
    with torch.no_grad():
        if INTERPOLATION and prompt_img2 is not None:
            # INTERPOLATION mode: use two images and create interpolated embeddings
            prompt_img2 = transform(prompt_img2).unsqueeze(0)
            prompt_imgs = torch.cat([prompt_img1, prompt_img2], dim=0)
            caption_embs = dino(prompt_imgs.to(device))
            caption_embs = torch.nn.functional.normalize(caption_embs, dim=-1).unsqueeze(1).unsqueeze(1)
            caption_emb1 = caption_embs[0]
            caption_emb2 = caption_embs[-1]
            weights = np.arange(0, 1, 1/bsz).tolist()
            caption_embs = [caption_emb2 * wei + caption_emb1 * (1-wei) for wei in weights]
            caption_embs = torch.stack(caption_embs).to(device)
        else:
            # INTERPRETATION mode: use only one image and create its embedding
            prompt_imgs = prompt_img1
            caption_emb = dino(prompt_imgs.to(device))
            caption_emb = torch.nn.functional.normalize(caption_emb, dim=-1).unsqueeze(1).unsqueeze(1)
            caption_embs = caption_emb.repeat(bsz, 1, 1, 1).to(device)
        
        bsz = caption_embs.shape[0]
        null_y = model.y_embedder.y_embedding[None].repeat(bsz, 1, 1)[:, None]
        z = torch.randn(1, 4, 32, 32, device=device).repeat(bsz, 1, 1, 1)
        model_kwargs = dict(mask=None)
        dpm_solver = DPMS_INTER(model.forward_with_dpmsolver,
                            condition=caption_embs,
                            uncondition=null_y,
                            cfg_scale=guidance_scale,
                            model_kwargs=model_kwargs)
        output = dpm_solver.sample(
                z,
                steps=num_inference_steps,
                order=2,
                skip_type="time_uniform",
                method=method)
        output = vae.decode(output / 0.18215).sample
        output = torch.clamp(output * 0.5 + 0.5, min=0, max=1).cpu()
        output = (
            make_grid(output, nrow=output.shape[0] // 3, padding=3, pad_value=1).permute(1, 2, 0).numpy() * 255
        ).astype(np.uint8)
        return output


if __name__ == "__main__":
    parser = simple_parsing.ArgumentParser(description="Lumos Image Interpolation Generation Demo")
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
    ## transform for vision encoder
    transform = [
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(224),  # Image.BICUBIC
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]

    transform = T.Compose(transform)
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
    
    if INFO_GRAD:
        # Print frozen modules for all models
        print_frozen_modules(vae, "VAE")
        print_frozen_modules(dino, "DINO Vision Encoder")
        print_frozen_modules(model, "LumosI2I Diffusion Model")
    
    # Load images from input folder
    input_folder = os.path.join(os.path.dirname(__file__), "input")
    image1_path = os.path.join(input_folder, "n02099712_2668.JPEG")
    image2_path = os.path.join(input_folder, "n02099712_8719.JPEG")
    
    # Load the images based on mode
    prompt_img1 = Image.open(image1_path)
    prompt_img2 = None
    
    if INTERPOLATION:
        # INTERPOLATION mode: load both images
        prompt_img2 = Image.open(image2_path)
        bsz = 12
        output_filename = "output_interpolation.png"
    elif INTERPRETATION:
        # INTERPRETATION mode: use only one image
        bsz = 1
        output_filename = "output_interpretation.png"
    else:
        raise ValueError("Either INTERPOLATION or INTERPRETATION must be True")
    
    # Generate images
    guidance_scale = 4.5
    num_inference_steps = 20
    seed = 10
    randomize_seed = True
    
    # All available methods
    # methods = ["multistep", "singlestep", "singlestep_fixed", "adaptive"]
    methods = ["adaptive"]
    
    output_folder = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_folder, exist_ok=True)
    
    # Generate and save outputs for each method
    for method in methods:
        print(f"\n{'='*60}")
        print(f"Generating with method: {method}")
        print(f"{'='*60}")
        
        try:
            output = generate(
                prompt_img1=prompt_img1,
                prompt_img2=prompt_img2,
                bsz=bsz,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                randomize_seed=randomize_seed,
                method=method
            )
            
            # Save output
            output_image = Image.fromarray(output)
            
            # Create method-specific filename
            base_name = output_filename.replace(".png", "")
            method_output_filename = f"{base_name}_{method}.png"
            output_path = os.path.join(output_folder, method_output_filename)
            
            if INTERPRETATION:
                # Create combined image with input and output side by side with labels
                create_combined_image_with_labels(prompt_img1, output_image, output_path)
            else:
                output_image.save(output_path)
            
            print(f"✓ Output saved to {output_path}")
            
        except Exception as e:
            print(f"✗ Error with method {method}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("All methods completed!")
    print(f"{'='*60}")
