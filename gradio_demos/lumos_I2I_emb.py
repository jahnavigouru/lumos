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
from helper import create_combined_image_with_labels, freeze_model_parameters, create_transform, get_sam_loss
from loss import DirectionLoss, CLIPLoss

src_txt = "brown dog"
trg_txt = "white dog"

def generate_perturbed_image(
    embeddings,
    guidance_scale=4.5,
    seed=10,
    enable_grad=True
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
            steps=20,
            order=2,
            skip_type="time_uniform",
            method="multistep",
            enable_grad=enable_grad)
    output = vae.decode(output / 0.18215).sample
    output = torch.clamp(output * 0.5 + 0.5, min=0, max=1)
    
    return output

def compute_clip_loss(
    original_image,
    generated_image,
    clip_model,
    processor,
    device=None,
    lambda_direction=1.0,
    variant_axes=None,
    l2_reg_weight=0.01,
    lambda_clip=2.0,
    lambda_l1=0.1,
    sam_checkpoint=None,
    sam_model_type="vit_h",
    sam_background_dir=None,
    sam_save_every: int = 1,
):
    x0 = original_image
    x = generated_image

    clip_loss_func = CLIPLoss(clip_model, processor, loss_type='cosine', lambda_direction=lambda_direction)
    clip_loss_value = clip_loss_func(x0, src_txt, x, trg_txt, device=device)
    
    loss_clip = (2 - clip_loss_value) / 2
    loss_clip = -torch.log(loss_clip)
    
    loss_l1 = None
    sam_loss = get_sam_loss(
        sam_checkpoint=sam_checkpoint,
        sam_model_type=sam_model_type,
        device=device,
        background_dir=sam_background_dir,
        checkpoints_root=os.path.dirname(project_root),
        save_every=sam_save_every,
    )
    if sam_loss is not None:
        try:
            loss_l1 = sam_loss(x0, x)
            print(f"SAM Loss value: {loss_l1.item():.6f}")
        except Exception as exc:
            print(f"[warn] SAMLoss failed, using L1Loss instead: {exc}")
            l1_loss_fn = nn.L1Loss()
            loss_l1 = l1_loss_fn(x0, x)
    else:
        l1_loss_fn = nn.L1Loss()
        loss_l1 = l1_loss_fn(x0, x)

    total_loss = lambda_clip * loss_clip
    if loss_l1 is not None:
        total_loss = total_loss + lambda_l1 * loss_l1
    
    if variant_axes is not None and l2_reg_weight > 0:
        l2_reg_loss = l2_reg_weight * torch.norm(variant_axes) ** 2
        total_loss = total_loss + l2_reg_loss
    
    return total_loss

def optimize_perturbed_image(
    prompt_img1,
    guidance_scale=4.5,
    seed=10,
    num_iterations=10,
    learning_rate=5e-4,
    lambda_clip=2.0,
    lambda_l1=0.1,
    l2_reg_weight=0.01,
    max_grad_norm=5.0,
    max_variant_norm=0.20,
    output_folder=None,
):
    vae, dino, transform, model = models["vae"], models["vision_encoder"], models["transform"], models["diffusion"]
    
    freeze_model_parameters(vae)
    freeze_model_parameters(dino)
    freeze_model_parameters(model)
    
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    freeze_model_parameters(clip_model)

    prompt_img1_tensor = transform(prompt_img1).unsqueeze(0)
    
    with torch.no_grad():
        caption_emb = dino(prompt_img1_tensor.to(device))
        caption_emb = torch.nn.functional.normalize(caption_emb, dim=-1).unsqueeze(1).unsqueeze(1)
    
    # Set random seed for deterministic variant_axes initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    variant_axes = nn.Parameter(torch.randn_like(caption_emb) * 0.01).to(device)
    
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
    print(f"Learning rate: {learning_rate}, lambda_clip: {lambda_clip}, lambda_l1: {lambda_l1}")
    print(f"l2_reg_weight: {l2_reg_weight}, max_grad_norm: {max_grad_norm}, max variant norm: {max_variant_norm}")
    print(f"{'='*60}\n")
    
    # Generate original image from original embedding for comparison
    with torch.no_grad():
        original_image = generate_perturbed_image(
            embeddings=caption_emb,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_grad=True
        )
        if output_folder is not None:
            iter_image_copy = original_image[0].clone().cpu()
            iter_image_np = (iter_image_copy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            iter_image_pil = Image.fromarray(iter_image_np)
            iter_image_path = os.path.join(iteration_folder, "iteration_000.png")
            iter_image_pil.save(iter_image_path)
    
    for iteration in range(num_iterations):
        optimizer.zero_grad()
        
        perturbed_emb = caption_emb + variant_axes
        perturbed_emb = torch.nn.functional.normalize(perturbed_emb, dim=-1)
        
        generated_image = generate_perturbed_image(
            embeddings=perturbed_emb,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_grad=True
        )
        
        total_loss = compute_clip_loss(
            original_image=original_image,
            generated_image=generated_image,
            clip_model=clip_model,
            processor=processor,
            device=device,
            variant_axes=variant_axes,
            l2_reg_weight=l2_reg_weight,
            lambda_clip=lambda_clip,
            lambda_l1=lambda_l1,
            sam_checkpoint=os.path.join(os.path.dirname(project_root), "checkpoints", "sam_vit_h_4b8939.pth"),
            sam_model_type="vit_h",
            sam_background_dir=os.path.join(output_folder, "sam_backgrounds") if output_folder else None,
            sam_save_every=10 if num_iterations > 100 else 1,
        )
        
        total_loss.backward()
        if max_grad_norm is not None:
            nn.utils.clip_grad_norm_([variant_axes], max_grad_norm)
        
        optimizer.step()
        # if max_variant_norm is not None:
        #     with torch.no_grad():
        #         current_norm = variant_axes.norm()
        #         if current_norm > max_variant_norm:
        #             variant_axes.mul_(max_variant_norm / (current_norm + 1e-6))
         
        loss_history.append(total_loss.item())
        print(f"Iteration {iteration + 1}/{num_iterations}: Total Loss = {total_loss.item():.6f}")
        
        # Only persist every 10th image if training is long-running (>100 iters) to save space
        should_record_iteration = output_folder is None or num_iterations <= 100 or (iteration + 1) % 10 == 0
        should_save_iteration = output_folder is not None and should_record_iteration
        with torch.no_grad():
            iter_image_copy = generated_image[0].clone().cpu()
            iter_image_np = (iter_image_copy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            iter_image_pil = Image.fromarray(iter_image_np)
            if should_record_iteration:
                iteration_images.append(iter_image_pil)
            if should_save_iteration:
                iter_image_path = os.path.join(iteration_folder, f"iteration_{iteration + 1:03d}.png")
                iter_image_pil.save(iter_image_path)
    
    with torch.no_grad():
        perturbed_emb = caption_emb + variant_axes
        print(f"Perturbed embedding statistics:")
        print(f"  Max value: {perturbed_emb.max().item():.6f}")
        print(f"  Min value: {perturbed_emb.min().item():.6f}")
        print(f"  Norm: {torch.norm(perturbed_emb).item():.6f}")
        generated_image = generate_perturbed_image(
            embeddings=perturbed_emb,
            guidance_scale=guidance_scale,
            seed=seed
        )
    
    return generated_image, variant_axes, loss_history, iteration_images


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

    print(f"\n{'='*60}")
    print("Optimizing variant_axes")
    print(f"{'='*60}")

    base_image_path = os.path.join(input_folder, "n02112350_520.JPEG")
    prompt_img1 = Image.open(base_image_path)

    generated_image, variant_axes, loss_history, iteration_images = optimize_perturbed_image(
        prompt_img1=prompt_img1,
        guidance_scale=4.5,
        seed=10,
        num_iterations=400,
        learning_rate=5e-4,
        lambda_clip=3.0,
        lambda_l1=0.1,
        l2_reg_weight=0.01,
        max_grad_norm=5.0,
        max_variant_norm=0.20,
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
