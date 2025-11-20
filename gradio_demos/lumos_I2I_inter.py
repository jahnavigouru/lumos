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

INTERPOLATION = True
INTERPRETATION = False
LEARN_IMG_PERTURBATION = False

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


class BlockMasking(object):
    def __init__(self, mask_ratio=0.40, block_size=32, avoid_neighbors=True):
        self.mask_ratio = mask_ratio
        self.block_size = block_size
        self.avoid_neighbors = avoid_neighbors

    def __call__(self, img):
        # img: tensor (C, H, W)
        C, H, W = img.shape
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size

        total_blocks = num_blocks_h * num_blocks_w
        num_mask = int(total_blocks * self.mask_ratio)

        mask = torch.ones((num_blocks_h, num_blocks_w), device=img.device, dtype=img.dtype)
        chosen = set()

        def has_neighbor(r, c):
            for rr, cc in chosen:
                if abs(rr - r) <= 1 and abs(cc - c) <= 1:
                    return True
            return False

        all_positions = torch.randperm(total_blocks, device=img.device)

        for idx in all_positions:
            if len(chosen) >= num_mask:
                break

            r = idx // num_blocks_w
            c = idx % num_blocks_w

            if self.avoid_neighbors and has_neighbor(int(r), int(c)):
                continue

            chosen.add((int(r), int(c)))

        for (r, c) in chosen:
            mask[r, c] = 0

        print(
            f"BlockMasking: mask_ratio={self.mask_ratio:.2f}, "
            f"total_blocks={total_blocks}, masked_blocks={len(chosen)}"
        )

        mask = mask.repeat_interleave(self.block_size, dim=0)
        mask = mask.repeat_interleave(self.block_size, dim=1)
        mask = mask.unsqueeze(0)  # (1,H,W)

        return img * mask


def add_gaussian_noise(img, mean=0.0, std=0.1):
    noise = torch.randn_like(img) * std + mean
    return torch.clamp(img + noise, 0.0, 1.0)

def random_crop_original_minus(img, shrink=50):
    w, h = img.size
    crop_size = max(1, min(w, h) - shrink)
    if crop_size <= 0:
        return img
    if crop_size >= min(w, h):
        return img
    max_left = w - crop_size
    max_top = h - crop_size
    left = random.randint(0, max_left) if max_left > 0 else 0
    top = random.randint(0, max_top) if max_top > 0 else 0
    return img.crop((left, top, left + crop_size, top + crop_size))

def create_transform(learn_img_perturbation=False):
    if not learn_img_perturbation:
        return [
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(224), 
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    else:
        mask_op = BlockMasking(mask_ratio=0.10, block_size=36)
        return [
            T.Lambda(lambda img: img.convert('RGB')),
            T.Lambda(lambda img: random_crop_original_minus(img, shrink=50)),
            # T.ColorJitter(brightness=0.6, contrast=0.1, saturation=0.1, hue=0.05),
            # T.RandomHorizontalFlip(p=0.5),
            T.Resize(224), 
            T.CenterCrop(224),
            T.ToTensor(),
            # T.Lambda(add_gaussian_noise),
            # T.Lambda(lambda img: mask_op(img)),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
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
    for param in model.parameters():
        param.requires_grad = False

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
            # test = nn.Parameter(torch.randn_like(caption_emb) * 0.01)
            # caption_emb = caption_emb + test
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

    input_folder = os.path.join(os.path.dirname(__file__), "input")
    image2_path = os.path.join(input_folder, "n02099712_8719.JPEG")
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

    prompt_img2 = None
    if INTERPOLATION:
        prompt_img2 = Image.open(image2_path)
        bsz = 12
        output_filename_base = "output_interpolation"
    elif INTERPRETATION:
        bsz = 1
        output_filename_base = "output_interpretation"
    else:
        raise ValueError("Either INTERPOLATION or INTERPRETATION must be True")

        guidance_scale = 4.5
        num_inference_steps = 20
        seed = 10
        randomize_seed = True
        methods = ["multistep"]

        def process_generation(prompt_img1, image_tag=None, target_folder=aug_folder):
            safe_tag = image_tag.replace(" ", "_") if image_tag else None
            image_label = safe_tag or "default input"
            base_name = output_filename_base if not safe_tag else f"{output_filename_base}_{safe_tag}"

            for method in methods:
                print(f"\n{'='*60}")
                print(f"Generating with method: {method} (image: {image_label})")
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

                    method_output_filename = f"{base_name}_{method}.png"
                    output_path = os.path.join(target_folder, method_output_filename)

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

                    print(f"✓ Output saved to {output_path} (image: {image_label})")

                except Exception as e:
                    print(f"✗ Error with method {method} (image: {image_label}): {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue

        if LEARN_IMG_PERTURBATION:
            for image_path in input_image_paths:
                prompt_img1 = Image.open(image_path)
                image_tag = os.path.splitext(os.path.basename(image_path))[0]
                process_generation(prompt_img1, image_tag=image_tag, target_folder=aug_folder)
                prompt_img1.close()
        else:
            prompt_img1 = Image.open(os.path.join(input_folder, "n02111889_7148.JPEG"))
            process_generation(prompt_img1, target_folder=aug_folder)
            prompt_img1.close()

        if prompt_img2 is not None:
            prompt_img2.close()

        print(f"\n{'='*60}")
        print("All methods completed!")
        print(f"{'='*60}")
