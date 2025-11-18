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
from transformers import CLIPModel, CLIPProcessor

INTERPOLATION = False
INTERPRETATION = True
INFO_GRAD = False
LEARN_IMG_PERTURBATION = True
LEARN_EMB_PERTURBATION = False
_CLIP_PROMPTS = [
    "a fluffy dog",
    "a black dog",
    "a dog with blue eyes",
    "a dog with long ears",
]

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

def create_transform(learn_img_perturbation=False):
    if not learn_img_perturbation:
        return [
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(224),  # Image.BICUBIC
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        return [
            T.Lambda(lambda img: img.convert('RGB')),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.RandomHorizontalFlip(p=0.5),
            T.Resize(224),  # Image.BICUBIC
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]


def tensor_to_display_image(tensor):
    """
    Convert a normalized tensor (C, H, W) back to a displayable PIL image.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    img = tensor.clone()
    img = img * std + mean
    img = torch.clamp(img, 0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img)

def freeze_model_parameters(model):
    """
    Freeze all parameters of a model by setting requires_grad=False.
    
    Args:
        model: PyTorch model to freeze
    """
    for param in model.parameters():
        param.requires_grad = False

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

def create_interpretation_grid(original_input, original_output, augmented_input=None, augmented_output=None, output_path=None):
    # Helper function to convert image to displayable format (no resizing)
    def to_displayable(img):
        if isinstance(img, Image.Image):
            return img
        elif isinstance(img, torch.Tensor):
            # Convert tensor to numpy array for matplotlib
            if img.dim() == 4:
                img = img.squeeze(0)
            # Images from generate() are in [0, 1] range, so multiply by 255
            img_np = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            return img_np
        else:
            # numpy array
            return img
    
    grid_items = [
        (original_input, 'Input Image'),
        (original_output, 'Output Image'),
    ]

    if augmented_input is not None and augmented_output is not None:
        grid_items.extend([
            (augmented_input, 'Augmented Image'),
            (augmented_output, 'Augmented Output'),
        ])

    rows = 2 if len(grid_items) > 2 else 1
    cols = 2
    figsize = (12, 6 * rows)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for ax, (img, title) in zip(axes_list, grid_items):
        ax.imshow(to_displayable(img))
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')

    for extra_ax in axes_list[len(grid_items):]:
        extra_ax.axis('off')
    
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
    method="multistep",
    INTERPRETATION=True,
    img_per=False
):
    seed = int(randomize_seed_fn(seed, randomize_seed))
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    vae, dino, model = models["vae"], models["vision_encoder"], models["diffusion"]
    
    # Create transform based on img_per argument
    cur_transform = T.Compose(create_transform(learn_img_perturbation=img_per))

    # Apply transform to prompt_img1 (always needed for DINO preprocessing) and save a displayable copy
    prompt_img1_tensor = cur_transform(prompt_img1)
    display_input_image = tensor_to_display_image(prompt_img1_tensor)
    prompt_img1 = prompt_img1_tensor.unsqueeze(0)
    
    with torch.no_grad():
        if INTERPOLATION and prompt_img2 is not None:
            # INTERPOLATION mode: use two images and create interpolated embeddings
            prompt_img2 = cur_transform(prompt_img2).unsqueeze(0)
            prompt_imgs = torch.cat([prompt_img1, prompt_img2], dim=0)
            caption_embs = dino(prompt_imgs.to(device))
            caption_embs = torch.nn.functional.normalize(caption_embs, dim=-1).unsqueeze(1).unsqueeze(1)
            caption_emb1 = caption_embs[0]
            caption_emb2 = caption_embs[-1]
            weights = np.arange(0, 1, 1/bsz).tolist()
            caption_embs = [caption_emb2 * wei + caption_emb1 * (1-wei) for wei in weights]
            caption_embs = torch.stack(caption_embs).to(device)
        else:
            # Always use original image for main output
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
                method=method,
                enable_grad=False)
        output = vae.decode(output / 0.18215).sample
        output = torch.clamp(output * 0.5 + 0.5, min=0, max=1).cpu()
        
        if INTERPRETATION:
            return display_input_image, output
        else:
            output = (
                make_grid(output, nrow=output.shape[0] // 3, padding=3, pad_value=1).permute(1, 2, 0).numpy() * 255
            ).astype(np.uint8)
            return output

def generate_perturbed_image(
    embeddings,
    guidance_scale=4.5,
    seed=10,
    enable_grad=False
):
    """
    Generate image from embeddings using DPMS_INTER and VAE decoder.
    
    Args:
        embeddings: Tensor of embeddings (bsz, 1, 1, embed_dim) to use as condition
        guidance_scale: Guidance scale for classifier-free guidance
        seed: Random seed for noise generation
        enable_grad: Whether to enable gradients for optimization (default: False)
        
    Returns:
        output: Generated images tensor (B, C, H, W) in range [0, 1]
    """
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
    target_index=0,
    bsz=1,
    guidance_scale=4.5,
    seed=10,
    num_iterations=10,
    learning_rate=0.01,
    l2_reg_weight=0.01,
    output_folder=None
):
    """
    Train variant_axes for num_iterations using CLIP loss and generate perturbed image.
    
    Args:
        prompt_img1: Input PIL image
        target_index: Index of target prompt in _CLIP_PROMPTS to maximize similarity with
        bsz: Batch size
        guidance_scale: Guidance scale for generation
        seed: Random seed
        num_iterations: Number of training iterations
        learning_rate: Learning rate for optimizer
        l2_reg_weight: Weight for L2 regularization
        output_folder: Optional folder path to save iteration images
        
    Returns:
        generated_image: Final generated image tensor (B, C, H, W) in range [0, 1]
        variant_axes: Optimized learnable parameter tensor (bsz, 1, 1, embed_dim)
        loss_history: List of loss values during training
        iteration_images: List of PIL Images from each iteration
    """
    vae, dino, transform, model = models["vae"], models["vision_encoder"], models["transform"], models["diffusion"]
    
    # Ensure all models are frozen
    freeze_model_parameters(vae)  # Freeze VAE encoder and decoder
    freeze_model_parameters(dino)  # Freeze DINO vision encoder
    freeze_model_parameters(model)  # Freeze diffusion model
    
    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    freeze_model_parameters(clip_model)  # Freeze CLIP model
    
    # Transform and get image embedding
    prompt_img1_tensor = transform(prompt_img1).unsqueeze(0)
    
    with torch.no_grad():
        # Generate embeddings from input image
        caption_emb = dino(prompt_img1_tensor.to(device))
        caption_emb = torch.nn.functional.normalize(caption_emb, dim=-1).unsqueeze(1).unsqueeze(1)
    
    # Create learnable variant_axes parameter (will be moved to device in generate_perturbed_image)
    variant_axes = nn.Parameter(torch.randn_like(caption_emb) * 0.01)
    
    # Setup optimizer (only optimize variant_axes)
    optimizer = torch.optim.Adam([variant_axes], lr=learning_rate)
    
    loss_history = []
    iteration_images = []  # Store images from each iteration
    
    # Create output folder for iteration images if provided
    if output_folder is not None:
        os.makedirs(output_folder, exist_ok=True)
        iteration_folder = os.path.join(output_folder, "iterations")
        os.makedirs(iteration_folder, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training variant_axes for {num_iterations} iterations")
    print(f"Target prompt: {_CLIP_PROMPTS[target_index]}")
    print(f"Learning rate: {learning_rate}, L2 reg weight: {l2_reg_weight}")
    print(f"{'='*60}\n")
    
    # Training loop
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        # Combine embeddings and variant_axes to create perturbed_emb (without normalization)
        perturbed_emb = caption_emb + variant_axes
        
        # Generate image using perturbed embeddings
        # Enable gradients for optimization
        generated_image = generate_perturbed_image(
            embeddings=perturbed_emb,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_grad=False  # Enable gradients to flow through diffusion process
        )
        
        # Compute CLIP loss
        total_loss, similarities = compute_clip_loss(
            generated_image=generated_image,
            variant_axes=variant_axes,
            clip_model=clip_model,
            processor=processor,
            target_index=target_index,
            l2_reg_weight=l2_reg_weight,
            device=device
        )
        
        # Backward pass
        total_loss.backward()
        
        # Check if gradients are flowing to variant_axes
        # Note: Gradients flow through: loss → image → VAE → diffusion → perturbed_emb → variant_axes_normalized → variant_axes
        # The normalization operation is differentiable, so gradients will flow through it
        grad_norm = torch.norm(variant_axes.grad) if variant_axes.grad is not None else torch.tensor(0.0)
        
        # Diagnostic: Check if variant_axes has gradients
        has_grad = variant_axes.grad is not None
        if not has_grad:
            print(f"WARNING: variant_axes has no gradients! Gradient flow may be broken.")
        
        # Store previous variant_axes before update (for tracking weight changes)
        prev_variant_axes = variant_axes.data.clone()
        
        optimizer.step()
        
        # Compute weight change after update
        weight_change = torch.norm(variant_axes.data - prev_variant_axes)
        weight_change_percent = (weight_change / (torch.norm(prev_variant_axes) + 1e-8)) * 100
        
        loss_history.append(total_loss.item())
        
        # Save image from this iteration
        # Make a copy of the generated image to avoid detaching the original
        with torch.no_grad():
            iter_image_copy = generated_image[0].clone().cpu()
            iter_image_np = (iter_image_copy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            iter_image_pil = Image.fromarray(iter_image_np)
            iteration_images.append(iter_image_pil)
            
            # Save to disk if output folder is provided
            if output_folder is not None:
                iter_image_path = os.path.join(iteration_folder, f"iteration_{iteration + 1:03d}.png")
                iter_image_pil.save(iter_image_path)
        
        # Print progress
        if (iteration + 1) % max(1, num_iterations // 5) == 0 or iteration == 0:
            print(f"Iteration {iteration + 1}/{num_iterations}:")
            print(f"  Total Loss: {total_loss.item():.4f}")
            print(f"  Similarities: {similarities[0].cpu().detach().numpy()}")
            print(f"  Gradient Norm: {grad_norm.item():.6f}")
            print(f"  Weight Change (L2 norm): {weight_change.item():.6f}")
            print(f"  Weight Change (%): {weight_change_percent.item():.4f}%")
            print()
    
    # Generate final image with optimized variant_axes
    with torch.no_grad():
        perturbed_emb = caption_emb + variant_axes
        generated_image = generate_perturbed_image(
            embeddings=perturbed_emb,
            guidance_scale=guidance_scale,
            seed=seed
        )
    
    return generated_image, variant_axes, loss_history, iteration_images

def compute_clip_loss(
    generated_image,
    variant_axes,
    clip_model,
    processor,
    target_index,
    l2_reg_weight=0.01,
    device=None
):
    """
    Compute CLIP loss using cross-entropy: maximize similarity with target prompt, plus L2 regularization.
    
    Args:
        generated_image: Generated image tensor (B, C, H, W) in range [0, 1]
        variant_axes: Learnable parameter tensor for L2 regularization
        clip_model: CLIP model
        processor: CLIP processor
        target_index: Index of target prompt in _CLIP_PROMPTS to maximize similarity with
        l2_reg_weight: Weight for L2 regularization on variant_axes
        device: torch device
        
    Returns:
        loss: Total loss value (scalar tensor)
        similarities: Tensor of similarities (B, num_prompts)
    """
    # Preprocess image for CLIP - resize to 224x224 and normalize
    # Resize to 224x224
    clip_image = torch.nn.functional.interpolate(
        generated_image, size=(224, 224), mode='bilinear', align_corners=False
    )
    # CLIP normalization
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
    clip_image = (clip_image - mean) / std
    
    # Get image features with gradients
    image_features = clip_model.get_image_features(pixel_values=clip_image)
    
    # Get text features without gradients
    text_inputs = processor(text=_CLIP_PROMPTS, return_tensors="pt", padding=True)
    text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
    with torch.no_grad():
        text_features = clip_model.get_text_features(
            input_ids=text_inputs['input_ids'], 
            attention_mask=text_inputs['attention_mask']
        )
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Get target text feature
    target_feat = text_features[target_index]
    
    # Negative cosine similarity (we want to maximize similarity, so minimize negative similarity)
    clip_loss = -torch.matmul(image_features, target_feat)
    clip_loss = clip_loss.mean()
    
    # Compute similarity matrix for return value (for diagnostics)
    similarities = torch.matmul(image_features, text_features.t())  # (B, num_prompts)
    
    # L2 regularization on variant_axes
    l2_reg = l2_reg_weight * torch.norm(variant_axes) ** 2
    
    # Total loss
    total_loss = clip_loss + l2_reg
    
    return total_loss, similarities


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
    transform = T.Compose(create_transform(learn_img_perturbation=LEARN_IMG_PERTURBATION))
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
    
    if LEARN_EMB_PERTURBATION:
        # LEARN_EMB_PERTURBATION mode: optimize variant_axes using CLIP loss
        print(f"\n{'='*60}")
        print("LEARN_EMB_PERTURBATION mode: Optimizing variant_axes")
        print(f"{'='*60}")
        
        target_index = 1  # Target prompt index (can be changed)
        
        # Create output folder for saving iteration images
        output_folder = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_folder, exist_ok=True)
        
        generated_image, variant_axes, loss_history, iteration_images = optimize_perturbed_image(
            prompt_img1=prompt_img1,
            target_index=target_index,
            bsz=1,
            guidance_scale=4.5,
            seed=10,
            num_iterations=10,
            learning_rate=0.01,
            l2_reg_weight=0.01,
            output_folder=output_folder  # Pass output folder to save iteration images
        )
        
        # Convert generated image to PIL and save
        generated_image_np = (generated_image[0].cpu().detach().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        output_image = Image.fromarray(generated_image_np)
        
        output_path = os.path.join(output_folder, "output_perturbed.png")
        
        # Create combined image with input and output
        create_combined_image_with_labels(prompt_img1, output_image, output_path)
        
        print(f"✓ Output saved to {output_path}")
        print(f"✓ Saved {len(iteration_images)} iteration images to {os.path.join(output_folder, 'iterations')}")
        print(f"Final loss: {loss_history[-1]:.4f}")
        print(f"{'='*60}\n")
        
    elif INTERPOLATION:
        # INTERPOLATION mode: load both images
        prompt_img2 = Image.open(image2_path)
        bsz = 12
        output_filename = "output_interpolation.png"
    elif INTERPRETATION:
        # INTERPRETATION mode: use only one image
        bsz = 1
        output_filename = "output_interpretation.png"
    else:
        raise ValueError("Either INTERPOLATION, INTERPRETATION, or LEARN_EMB_PERTURBATION must be True")
    
    if not LEARN_EMB_PERTURBATION:
        # Generate images
        guidance_scale = 4.5
        num_inference_steps = 20
        seed = 10
        randomize_seed = True
        
        # All available methods
        # methods = ["multistep", "singlestep", "singlestep_fixed", "adaptive"]
        methods = ["multistep"]
        
        output_folder = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate and save outputs for each method
        for method in methods:
            print(f"\n{'='*60}")
            print(f"Generating with method: {method}")
            print(f"{'='*60}")
            
            try:
                if INTERPRETATION and LEARN_IMG_PERTURBATION:
                    original_input, original_output = generate(
                        prompt_img1=prompt_img1,
                        prompt_img2=prompt_img2,
                        bsz=bsz,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        randomize_seed=randomize_seed,
                        method=method,
                        INTERPRETATION=INTERPRETATION,
                        img_per=False
                    )

                    augmented_input, augmented_output = generate(
                        prompt_img1=prompt_img1,
                        prompt_img2=prompt_img2,
                        bsz=bsz,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        randomize_seed=randomize_seed,
                        method=method,
                        INTERPRETATION=INTERPRETATION,
                        img_per=True
                    )
                else:
                    result_input, result_output = generate(
                        prompt_img1=prompt_img1,
                        prompt_img2=prompt_img2,
                        bsz=bsz,
                        guidance_scale=guidance_scale,
                        num_inference_steps=num_inference_steps,
                        seed=seed,
                        randomize_seed=randomize_seed,
                        method=method,
                        INTERPRETATION=INTERPRETATION,
                        img_per=False
                    )
                
                # Create method-specific filename
                base_name = output_filename.replace(".png", "")
                method_output_filename = f"{base_name}_{method}.png"
                output_path = os.path.join(output_folder, method_output_filename)
                
                if INTERPRETATION:
                    if LEARN_IMG_PERTURBATION:
                        create_interpretation_grid(
                            original_input=original_input,
                            original_output=original_output,
                            augmented_input=augmented_input,
                            augmented_output=augmented_output,
                            output_path=output_path,
                        )
                    else:
                        create_interpretation_grid(
                            original_input=result_input,
                            original_output=result_output,
                            augmented_input=None,
                            augmented_output=None,
                            output_path=output_path,
                        )
                else:
                    if LEARN_IMG_PERTURBATION:
                        output = (
                            make_grid(augmented_output, nrow=augmented_output.shape[0] // 3, padding=3, pad_value=1).permute(1, 2, 0).numpy() * 255
                        ).astype(np.uint8)
                        output_image = Image.fromarray(output)
                    else:
                        output = (
                            make_grid(result_output, nrow=result_output.shape[0] // 3, padding=3, pad_value=1).permute(1, 2, 0).numpy() * 255
                        ).astype(np.uint8)
                        output_image = Image.fromarray(output)
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
