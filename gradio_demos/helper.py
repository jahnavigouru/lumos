import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image


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
        return [
            T.Lambda(lambda img: img.convert('RGB')),
            T.Resize(224), 
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ]


def freeze_model_parameters(model):
    for param in model.parameters():
        param.requires_grad = False

def create_combined_image_with_labels(input_image, output_image, output_path):
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

def save_iteration_image(image_tensor, iteration_folder, filename, print_message=None):
    """
    Save an image tensor to disk as a PNG file in the iteration folder.
    
    Args:
        image_tensor: torch.Tensor - Image tensor of shape (C, H, W) or (B, C, H, W)
        iteration_folder: str - Path to the iteration folder where image will be saved
        filename: str - Filename to save the image as (e.g., "iteration_000.png")
        print_message: str or None - Optional message to print after saving
    """
    os.makedirs(iteration_folder, exist_ok=True)
    
    # Handle batch dimension if present
    if image_tensor.dim() == 4:
        image_tensor = image_tensor[0]
    
    # Convert tensor to PIL Image
    image_copy = image_tensor.clone().cpu()
    image_np = (image_copy.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    
    # Save image
    image_path = os.path.join(iteration_folder, filename)
    image_pil.save(image_path)
    
    if print_message is not None:
        print(print_message)
    
    return image_pil

