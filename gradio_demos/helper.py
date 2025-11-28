import os
from datetime import datetime
from typing import Any, Dict, Union

import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
import PIL
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

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

def plot_all_trajectories(trajectory_map, output_dir=None, methods=['PCA', 'TSNE', 'UMAP']):
    """
    Plot all trajectories from trajectory_map in a single graph for each method.
    Each trajectory will have different colors, labels, and start/end markers.
    
    Args:
        trajectory_map: dict - dictionary with keys as trajectory names and values as embeddings tensors
        output_dir: str - directory to save plots (default: plots folder)
        methods: list - list of methods to use for dimensionality reduction ['PCA', 'TSNE', 'UMAP']
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"plot_all_trajectories: Received {len(trajectory_map)} trajectories")
    for traj_name in trajectory_map.keys():
        print(f"  - {traj_name}")
    
    # Prepare all embeddings for combined dimensionality reduction
    all_embeddings_list = []
    trajectory_labels = []
    trajectory_indices = []
    trajectory_names = []  # Store names in order
    
    for traj_name, embeddings in trajectory_map.items():
        # Reshape embeddings to [bsz, 768] if needed
        if embeddings.dim() == 4:
            embeddings_flat = embeddings.squeeze(1).squeeze(1)  # [bsz, 768]
        elif embeddings.dim() == 2:
            embeddings_flat = embeddings
        else:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")
        
        embeddings_np = embeddings_flat.detach().cpu().numpy()
        all_embeddings_list.append(embeddings_np)
        trajectory_labels.extend([traj_name] * embeddings_np.shape[0])
        start_idx = len(trajectory_labels) - embeddings_np.shape[0]
        end_idx = len(trajectory_labels)
        trajectory_indices.append((start_idx, end_idx))
        trajectory_names.append(traj_name)  # Store name in same order
    
    # Combine all embeddings
    all_embeddings = np.vstack(all_embeddings_list)
    
    # Plot for each method
    for method in methods:
        method_upper = method.upper()
        
        if method_upper == 'PCA':
            print("Computing PCA for all trajectories...")
            reducer = PCA(n_components=2)
            embeddings_reduced = reducer.fit_transform(all_embeddings)
            xlabel = f'PC1 ({reducer.explained_variance_ratio_[0]:.2%} variance)'
            ylabel = f'PC2 ({reducer.explained_variance_ratio_[1]:.2%} variance)'
            title = 'All Trajectories - PCA'
            filename = 'all_trajectories_pca.png'
            
        elif method_upper in ['TSNE', 'T-SNE']:
            print("Computing t-SNE for all trajectories (this may take a while)...")
            n_samples = all_embeddings.shape[0]
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples-1), n_iter=1000)
            embeddings_reduced = reducer.fit_transform(all_embeddings)
            xlabel = 't-SNE Dimension 1'
            ylabel = 't-SNE Dimension 2'
            title = 'All Trajectories - t-SNE'
            filename = 'all_trajectories_tsne.png'
            
        elif method_upper == 'UMAP':
            if not UMAP_AVAILABLE:
                print(f"Warning: UMAP not available, skipping {method}")
                continue
            print("Computing UMAP for all trajectories...")
            n_samples = all_embeddings.shape[0]
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, n_samples-1))
            embeddings_reduced = reducer.fit_transform(all_embeddings)
            xlabel = 'UMAP Dimension 1'
            ylabel = 'UMAP Dimension 2'
            title = 'All Trajectories - UMAP'
            filename = 'all_trajectories_umap.png'
        else:
            print(f"Unknown method: {method}, skipping")
            continue
        
        # Create plot with all trajectories
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Use different colors for each trajectory
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectory_names)))
        color_map = {name: colors[i] for i, name in enumerate(trajectory_names)}
        
        print(f"Plotting {len(trajectory_names)} trajectories for {method}")
        
        # Plot each trajectory
        for traj_name, (start_idx, end_idx) in zip(trajectory_names, trajectory_indices):
            print(f"  Plotting trajectory: {traj_name}, indices: {start_idx} to {end_idx}")
            traj_embeddings = embeddings_reduced[start_idx:end_idx]
            color = color_map[traj_name]
            
            # Plot trajectory line (connect all points in order)
            ax.plot(traj_embeddings[:, 0], traj_embeddings[:, 1], '-', 
                   color=color, alpha=0.6, linewidth=2.5, label=traj_name, zorder=2)
            
            # Plot all points with correct color array
            n_points = traj_embeddings.shape[0]
            ax.scatter(traj_embeddings[:, 0], traj_embeddings[:, 1], 
                      c=[color] * n_points, s=50, alpha=0.7, edgecolors='black', 
                      linewidths=0.5, zorder=3)
            
            # Mark start point
            ax.scatter(traj_embeddings[0, 0], traj_embeddings[0, 1], 
                      c='red', s=200, marker='o', edgecolors='black', 
                      linewidths=2, zorder=5, label='Start' if traj_name == trajectory_names[0] else '')
            
            # Mark end point
            ax.scatter(traj_embeddings[-1, 0], traj_embeddings[-1, 1], 
                      c='blue', s=200, marker='s', edgecolors='black', 
                      linewidths=2, zorder=5, label='End' if traj_name == trajectory_names[0] else '')
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✓ {method} combined plot saved to {output_path}")


def save_mask(
    result: Dict[str, Any],
    original_image: Union[np.ndarray, PIL.Image.Image] = None,
    output_dir: str = None,
    iteration: int = None
) -> None:
    """
    Save the calculated masks to the output/sam folder as a matplotlib visualization.
    
    Args:
        result: Dictionary containing 'foreground_mask' and 'background_mask' from calculate_mask()
        original_image: Original image to display alongside masks. Can be numpy array or PIL Image.
        output_dir: Optional custom output directory. If None, uses output/sam relative to script location.
        iteration: Optional iteration number to include in the filename.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output", "sam")
    
    os.makedirs(output_dir, exist_ok=True)
    
    foreground_mask = result["foreground_mask"]
    background_mask = result["background_mask"]
    
    # Generate unique timestamp for filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if iteration is not None:
        filename_suffix = f"iter_{iteration:04d}_{timestamp}"
    else:
        filename_suffix = timestamp
    
    # Convert original image to numpy array if needed
    if original_image is not None:
        if isinstance(original_image, PIL.Image.Image):
            original_image_np = np.array(original_image)
        elif isinstance(original_image, np.ndarray):
            original_image_np = original_image.copy()
        else:
            original_image_np = None
    else:
        original_image_np = None
    
    # Create matplotlib figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot original image
    if original_image_np is not None:
        axes[0].imshow(original_image_np)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, "No original image", ha='center', va='center')
        axes[0].set_title("Original Image")
        axes[0].axis('off')
    
    # Plot foreground mask
    foreground_mask_uint8 = (foreground_mask.astype(np.uint8) * 255)
    axes[1].imshow(foreground_mask_uint8, cmap='gray')
    axes[1].set_title("Foreground Mask")
    axes[1].axis('off')
    
    # Plot background mask
    background_mask_uint8 = (background_mask.astype(np.uint8) * 255)
    axes[2].imshow(background_mask_uint8, cmap='gray')
    axes[2].set_title("Background Mask")
    axes[2].axis('off')
    
    # Save the combined visualization
    output_path = os.path.join(output_dir, f"mask_visualization_{filename_suffix}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_augmented_images(
    augmented_input: torch.Tensor,
    dist: torch.Tensor,
    batch_size: int,
    output_dir: str = None,
    iteration: int = None
) -> None:
    """
    Save augmented images to the output/sam folder with their clip losses.
    
    Args:
        augmented_input: Tensor containing all augmented images [num_augmented, C, H, W]
        dist: Tensor containing clip loss distances for each augmented image
        batch_size: Batch size of the original input
        output_dir: Optional custom output directory. If None, uses output/sam relative to script location.
        iteration: Optional iteration number to include in the filename.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output", "sam")
    
    os.makedirs(output_dir, exist_ok=True)
    
    num_augmented = augmented_input.shape[0]
    
    # Print clip loss for each augmented image
    if iteration is not None:
        print(f"\nIteration {iteration} - Clip losses for each augmented image:")
    else:
        print(f"\nClip losses for each augmented image:")
    
    for i in range(num_augmented):
        aug_idx = i
        aug_type = "non-augmented" if aug_idx < batch_size else f"augmented-{aug_idx - batch_size + 1}"
        clip_loss_individual = dist[aug_idx].item()
        print(f"  {aug_type}: {clip_loss_individual:.6f}")
        
        # Save individual augmented image
        aug_image_tensor = augmented_input[aug_idx].detach().cpu()
        # Convert from [-1, 1] range to [0, 1] range
        aug_image_np = ((aug_image_tensor.permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        aug_image_np = (aug_image_np * 255).astype(np.uint8)
        aug_image_pil = PIL.Image.fromarray(aug_image_np)
        
        # Create filename with iteration and augmentation type
        if iteration is not None:
            filename = f"augmented_iter_{iteration:04d}_{aug_type}.png"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"augmented_{aug_type}_{timestamp}.png"
        
        aug_image_path = os.path.join(output_dir, filename)
        aug_image_pil.save(aug_image_path)


def save_background_comparison_grid(
    x_tensor: torch.Tensor,
    masked_bg_tensor: torch.Tensor,
    init_image: torch.Tensor,
    init_bg_reference: torch.Tensor,
    output_dir: str = None,
    iteration: int = None
) -> None:
    """
    Save a grid comparison of input image, masked background, original image, and background reference.
    
    Args:
        x_tensor: Current input image tensor [B, C, H, W]
        masked_bg_tensor: Background portion of current image [B, C, H, W]
        init_image: Original full image tensor [B, C, H, W]
        init_bg_reference: Background reference (init_image_bg or init_image) [B, C, H, W]
        output_dir: Optional custom output directory. If None, uses output/sam relative to script location.
        iteration: Optional iteration number to include in the filename.
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), "output", "sam")
    
    os.makedirs(output_dir, exist_ok=True)
    
    def tensor_to_numpy(tensor):
        """Convert tensor from [-1, 1] range to [0, 255] uint8 numpy array."""
        if tensor.dim() == 4:
            tensor = tensor[0]  # Take first batch
        tensor_np = tensor.detach().cpu().permute(1, 2, 0).numpy()
        tensor_np = ((tensor_np + 1) / 2).clip(0, 1)
        tensor_np = (tensor_np * 255).astype(np.uint8)
        return tensor_np
    
    # Convert all tensors to numpy arrays
    x_np = tensor_to_numpy(x_tensor)
    masked_bg_np = tensor_to_numpy(masked_bg_tensor)
    init_image_np = tensor_to_numpy(init_image)
    init_bg_ref_np = tensor_to_numpy(init_bg_reference)
    
    # Create matplotlib figure with 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Top left: Input image (x_tensor)
    axes[0, 0].imshow(x_np)
    axes[0, 0].set_title("Input Image (x_tensor)")
    axes[0, 0].axis('off')
    
    # Top right: Masked background (masked_bg_tensor)
    axes[0, 1].imshow(masked_bg_np)
    axes[0, 1].set_title("Masked Background (masked_bg_tensor)")
    axes[0, 1].axis('off')
    
    # Bottom left: Original image (init_image)
    axes[1, 0].imshow(init_image_np)
    axes[1, 0].set_title("Original Image (init_image)")
    axes[1, 0].axis('off')
    
    # Bottom right: Background reference (init_bg_reference)
    axes[1, 1].imshow(init_bg_ref_np)
    axes[1, 1].set_title("Background Reference (init_bg_reference)")
    axes[1, 1].axis('off')
    
    # Create filename
    if iteration is not None:
        filename = f"background_comparison_iter_{iteration:04d}.png"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"background_comparison_{timestamp}.png"
    
    output_path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
