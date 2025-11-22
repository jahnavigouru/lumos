import os
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loss import SAMLoss
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

_SAM_CKPT_FILENAMES = {
    "vit_h": "sam_vit_h_4b8939.pth",
    "vit_l": "sam_vit_l_0b3195.pth",
    "vit_b": "sam_vit_b_01ec64.pth",
}
_SAM_LOSS_CACHE = {}


def get_sam_loss(
    sam_checkpoint=None,
    sam_model_type="vit_h",
    device=None,
    background_dir=None,
    checkpoints_root=None,
    save_every: int = 1,
):
    """
    Lazily initialize and cache SAMLoss.
    - sam_checkpoint: explicit checkpoint path; if None, use checkpoints_root + filename.
    - sam_model_type: one of vit_h/vit_l/vit_b.
    - device: torch device string; defaults to CUDA if available.
    - background_dir: optional dir to save background-only images from SAMLoss.
    - checkpoints_root: base dir containing checkpoints/; defaults to repo root.
    """
    base_root = (
        checkpoints_root
        if checkpoints_root is not None
        else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    )

    if sam_checkpoint is None:
        filename = _SAM_CKPT_FILENAMES.get(sam_model_type, _SAM_CKPT_FILENAMES["vit_h"])
        sam_checkpoint = os.path.join(base_root, "checkpoints", filename)

    key = (sam_model_type, sam_checkpoint)
    if key in _SAM_LOSS_CACHE:
        return _SAM_LOSS_CACHE[key]

    if not os.path.exists(sam_checkpoint):
        print(f"[warn] SAM checkpoint not found at {sam_checkpoint}; skipping SAM loss.")
        return None
    try:
        sam_loss = SAMLoss(
            model_type=sam_model_type,
            checkpoint=sam_checkpoint,
            device=device,
            save_background_dir=background_dir,
            save_every=save_every,
        )
        _SAM_LOSS_CACHE[key] = sam_loss
        return sam_loss
    except Exception as exc:
        print(f"[warn] Failed to initialize SAMLoss ({exc}); skipping SAM loss.")
        return None

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
