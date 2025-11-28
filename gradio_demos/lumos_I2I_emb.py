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
from helper import create_combined_image_with_labels, freeze_model_parameters, create_transform
from sam import Segmentation

src_txt = "brown dog"
trg_txt = "a picture of a black dog"
query_txt = "dog"

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
    
    # Ensure VAE decode preserves gradients
    if enable_grad and embeddings.requires_grad:
        # VAE should preserve gradients through forward pass even in eval mode
        # Only check gradients if embeddings actually require them
        vae_input = output / 0.18215
        # Check if vae_input has gradients (from dpm_solver output)
        if not vae_input.requires_grad:
            raise RuntimeError(
                f"VAE input does not require gradients. "
                f"This means gradients were lost in dpm_solver.sample(). "
                f"embeddings.requires_grad: {embeddings.requires_grad}"
            )
        vae_output = vae.decode(vae_input)
        output = vae_output.sample
        # Verify output has gradients after VAE decode
        if not output.requires_grad:
            raise RuntimeError(
                f"VAE output does not require gradients after decode. "
                f"vae_input.requires_grad: {vae_input.requires_grad}, "
                f"vae.eval(): {not vae.training}"
            )
    else:
        # If enable_grad=False or embeddings don't require grad, decode without gradient tracking
        with torch.no_grad():
            output = vae.decode(output / 0.18215).sample
    
    output = torch.clamp(output * 0.5 + 0.5, min=0, max=1)
    
    return output

# Global Segmentation instance (initialized on first use)
_segmentation_instance = None

def compute_total_loss(
    original_image,
    generated_image,
    device=None,
    clip_guidance_lambda=1000.0,
    range_lambda=50.0,
    lpips_sim_lambda=1000.0,
    l2_sim_lambda=10000.0,
    sam_checkpoint=None,
    sam_model_type="vit_h",
    query_text=None,
    target_text=None,
    iteration=None,
):
    global _segmentation_instance
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if _segmentation_instance is None:
        if sam_checkpoint is None:
            sam_checkpoint = os.path.join(os.path.dirname(project_root), "checkpoints", "sam_vit_h_4b8939.pth")
        _segmentation_instance = Segmentation(
            checkpoint_path=os.path.dirname(sam_checkpoint),
            checkpoint_name=os.path.basename(sam_checkpoint),
            model_type=sam_model_type,
            device=str(device)
        )
    
    if query_text is None:
        query_text = query_txt
    if target_text is None:
        target_text = trg_txt
    
    if _segmentation_instance.init_image is None:
        _segmentation_instance.set_init_image(original_image, query_text=query_text)
    
    # Pass tensor directly to preserve gradients
    total_loss = _segmentation_instance.calculate_total_loss(
        image=generated_image,
        query_text=query_text,
        target_text=target_text,
        clip_guidance_lambda=clip_guidance_lambda,
        range_lambda=range_lambda,
        lpips_sim_lambda=lpips_sim_lambda,
        l2_sim_lambda=l2_sim_lambda,
        iteration=iteration,
    )
    
    return total_loss

def optimize_perturbed_image(
    prompt_img1,
    guidance_scale=4.5,
    seed=10,
    num_iterations=10,
    learning_rate=5e-4,
    clip_guidance_lambda=1000.0,
    range_lambda=50.0,
    lpips_sim_lambda=1000.0,
    l2_sim_lambda=10000.0,
    output_folder=None,
):
    vae, dino, transform, model = models["vae"], models["vision_encoder"], models["transform"], models["diffusion"]
    
    freeze_model_parameters(vae)
    freeze_model_parameters(dino)
    freeze_model_parameters(model)

    prompt_img1_tensor = transform(prompt_img1).unsqueeze(0)
    
    with torch.no_grad():
        caption_emb = dino(prompt_img1_tensor.to(device))
        caption_emb = torch.nn.functional.normalize(caption_emb, dim=-1).unsqueeze(1).unsqueeze(1)
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    variant_axes = nn.Parameter(torch.randn_like(caption_emb) * 0.01).to(device)
    variant_axes.requires_grad = True  # Explicitly ensure variant_axes requires gradients
    
    # Verify that only variant_axes requires gradients
    trainable_params = sum(1 for p in [variant_axes] if p.requires_grad)
    print(f"Number of trainable parameters: {trainable_params} (should be 1 for variant_axes)")
    
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
    print(f"clip_guidance_lambda: {clip_guidance_lambda}, range_lambda: {range_lambda}")
    print(f"lpips_sim_lambda: {lpips_sim_lambda}, l2_sim_lambda: {l2_sim_lambda}")
    print(f"{'='*60}\n")
    
    with torch.no_grad():
        original_image = generate_perturbed_image(
            embeddings=caption_emb,
            guidance_scale=guidance_scale,
            seed=seed,
            enable_grad=False  # original_image doesn't need gradients
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
        
        # Verify generated_image has gradients
        if not generated_image.requires_grad:
            raise RuntimeError(
                f"generated_image does not require gradients after VAE decode. "
                f"This breaks the computation graph. "
                f"perturbed_emb.requires_grad: {perturbed_emb.requires_grad}, "
                f"variant_axes.requires_grad: {variant_axes.requires_grad}"
            )
        
        total_loss = compute_total_loss(
            original_image=original_image,
            generated_image=generated_image,
            device=device,
            clip_guidance_lambda=clip_guidance_lambda,
            range_lambda=range_lambda,
            lpips_sim_lambda=lpips_sim_lambda,
            l2_sim_lambda=l2_sim_lambda,
            sam_checkpoint=os.path.join(os.path.dirname(project_root), "checkpoints", "sam_vit_h_4b8939.pth"),
            sam_model_type="vit_h",
            query_text=query_txt,
            target_text=trg_txt,
            iteration=iteration,
        )
        
        # Verify total_loss has gradients before backward
        if not total_loss.requires_grad:
            raise RuntimeError(
                f"total_loss does not require gradients. "
                f"generated_image.requires_grad={generated_image.requires_grad}, "
                f"total_loss.grad_fn={total_loss.grad_fn}"
            )
        
        total_loss.backward()
        optimizer.step()
        
        loss_history.append(total_loss.item())
        print(f"Iteration {iteration + 1}/{num_iterations}: Total Loss = {total_loss.item():.6f}")
        
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

    base_image_path = os.path.join(input_folder, "image.png")
    prompt_img1 = Image.open(base_image_path)

    generated_image, variant_axes, loss_history, iteration_images = optimize_perturbed_image(
        prompt_img1=prompt_img1,
        guidance_scale=4.5,
        seed=10,
        num_iterations=1500,
        learning_rate=5e-4,
        clip_guidance_lambda=1000.0,
        range_lambda=50.0,
        lpips_sim_lambda=1000.0,
        l2_sim_lambda=10000.0,
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
