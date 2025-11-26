import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    _SAM_AVAILABLE = True
except Exception:
    # Keep optional dependency truly optional; raise with a clear error when used.
    _SAM_AVAILABLE = False

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


class SAMLoss(torch.nn.Module):
    """
    Compute L1 loss only over background pixels, where background is defined as the
    inverse of the largest SAM mask (foreground) for each image.
    """

    def __init__(
        self,
        sam_model=None,
        model_type: str = "vit_h",
        checkpoint: str = None,
        device: str = None,
        save_background_dir: str = None,
        save_every: int = 1,
        clip_model=None,
        clip_processor=None,
    ):
        super().__init__()
        if not _SAM_AVAILABLE:
            raise ImportError("segment_anything is required for SAMLoss (pip install segment-anything)")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if sam_model is None:
            if checkpoint is None:
                raise ValueError("Provide either an initialized sam_model or a checkpoint path to load SAM.")
            sam_model = sam_model_registry[model_type](checkpoint=checkpoint)

        self.sam = sam_model.to(self.device)
        self.sam.eval()
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)
        self.l1 = nn.L1Loss(reduction="mean")
        self.save_background_dir = save_background_dir
        self._save_idx = 0
        self.save_every = max(1, int(save_every))
        # Cache for original image mask (img_a is always the original image)
        self._cached_original_mask = None
        self._cached_original_num_masks = None
        # CLIP components for mask scoring
        self.clip_model = clip_model
        self.clip_processor = clip_processor

    def best_mask(
        self,
        img,
        text="dog",
        min_area_ratio: float = 0.01,
        area_bonus: float = 0.5,
        score_floor: float = None,
    ):
        if self.clip_model is None or self.clip_processor is None:
            raise ValueError("CLIP model and processor must be provided to SAMLoss for best_mask method")
        
        img_np = img.detach().permute(1, 2, 0).cpu().numpy()
        masks = self.mask_generator.generate(img_np)
        num_masks = len(masks)
        if num_masks == 0:
            return None
        
        h, w, _ = img_np.shape
        total_area = float(h * w)
        # Prepare text for CLIP
        text_inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True)
        text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(
                input_ids=text_inputs['input_ids'],
                attention_mask=text_inputs['attention_mask']
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        best_score = float("-inf")
        best_mask = None
        best_area_ratio = 0.0
        
        # Convert image to tensor if needed
        img_tensor = img.to(self.device)
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
        
        # Compare each mask with CLIP score
        for mask_data in masks:
            mask = mask_data["segmentation"].astype(bool)
            area_ratio = mask.sum() / total_area
            if area_ratio < min_area_ratio:
                # Drop tiny masks (e.g., just a nose) that dominate by CLIP score but are not foreground
                continue
            
            # Create masked image (set background to 0)
            masked_img_np = img_np.copy()
            masked_img_np[~mask] = 0
            
            # Convert to tensor and prepare for CLIP
            masked_img_tensor = torch.from_numpy(masked_img_np).permute(2, 0, 1).float().to(self.device)
            if masked_img_tensor.max() <= 1.5:
                masked_img_tensor = masked_img_tensor / 255.0
            masked_img_tensor = masked_img_tensor.unsqueeze(0)  # Add batch dimension
            
            # Prepare image for CLIP
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=self.device).view(1, 3, 1, 1)
            masked_img_tensor = torch.nn.functional.interpolate(masked_img_tensor, size=(224, 224), mode='bilinear', align_corners=False)
            masked_img_tensor = (masked_img_tensor - mean) / std
            
            # Get image features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=masked_img_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity (CLIP score)
            similarity = (image_features * text_features).sum(dim=-1).item()
            # Encourage larger masks that still match the text
            score = similarity + area_bonus * area_ratio
            
            if score > best_score:
                best_score = score
                best_mask = mask
                best_area_ratio = area_ratio
        
        if score_floor is not None and best_score < score_floor:
            print(f"Best mask rejected (score {best_score:.4f} below floor {score_floor:.4f})")
            return None, best_score, best_area_ratio

        print(f"Best mask selected with score: {best_score:.4f} (area_ratio bonus applied, area_ratio={best_area_ratio:.3f}) for text: '{text}'")
        return best_mask, best_score, best_area_ratio

    def _largest_mask(self, img: torch.Tensor, return_count=False):
        """
        Generate the largest SAM mask for a single image tensor shaped (C, H, W).
        Returns a numpy boolean array of shape (H, W) or None if no masks are found.
        If return_count=True, returns (mask, num_masks) tuple.
        """
        img_np = img.detach().permute(1, 2, 0).cpu().numpy()
        masks = self.mask_generator.generate(img_np)
        num_masks = len(masks)
        if num_masks == 0:
            return (None, 0) if return_count else None
        largest = max(masks, key=lambda m: m.get("area", 0))
        mask = largest["segmentation"].astype(bool)
        return (mask, num_masks) if return_count else mask

    def _save_background_image(self, img: torch.Tensor, background_mask, label: str):
        """Save a background-only image for inspection if a directory was provided."""
        if self.save_background_dir is None:
            return
        os.makedirs(self.save_background_dir, exist_ok=True)
        # img expected on any device, shape (C,H,W), values assumed in [0,1] or [0,255]
        img_np = img.detach().cpu()
        if img_np.max() <= 1.5:
            img_np = (img_np * 255.0).clamp(0, 255)
        img_np = img_np.byte().permute(1, 2, 0).numpy()
        bg_mask = background_mask.astype(bool)
        bg_img = img_np.copy()
        bg_img[~bg_mask] = 0
        fname = f"{label}_background_{self._save_idx:04d}.png"
        Image.fromarray(bg_img).save(os.path.join(self.save_background_dir, fname))

    def _background_l1(self, img_a: torch.Tensor, img_b: torch.Tensor, text="dog") -> torch.Tensor:
        # Use cached mask for original image (img_a) if available, otherwise compute and cache it
        # The mask is computed ONLY ONCE for img_a and reused for all iterations
        if self._cached_original_mask is None:
            mask_a, num_masks_a = self._largest_mask(img_a, return_count=True)
            self._cached_original_mask = mask_a
            self._cached_original_num_masks = num_masks_a
            print(f"SAMLoss: Computing original image mask (img_a) - found {num_masks_a} masks (cached for future iterations)")
        else:
            mask_a = self._cached_original_mask
            num_masks_a = self._cached_original_num_masks
            print(f"SAMLoss: Using cached original image mask (img_a) - {num_masks_a} masks")
        
        # Prefer CLIP-guided best_mask for generated image (img_b) so the dog stays foreground.
        mask_b = None
        num_masks_b = None
        if self.clip_model is not None and self.clip_processor is not None:
            try:
                mask_b, best_score, best_area_ratio = self.best_mask(
                    img_b,
                    text=text,
                    min_area_ratio=0.15,  # demand a reasonably sized dog mask
                    area_bonus=1.0,
                    score_floor=None,  # optional: set to e.g., 0.05 to reject weak matches
                )
                num_masks_b = -1  # indicator for "CLIP best mask"
                print(f"SAMLoss: Using best_mask for img_b (generated) with text: '{text}', area_ratio={best_area_ratio:.3f}, score={best_score:.4f}")
            except Exception as exc:
                print(f"[warn] SAMLoss best_mask failed ({exc}); falling back to largest mask.")
        # Fallback to largest mask if CLIP not available or failed
        if mask_b is None:
            mask_b, num_masks_b = self._largest_mask(img_b, return_count=True)
            print(f"SAMLoss: img_b (generated) has {num_masks_b} masks (using largest mask)")

        # Fallback to full L1 if either mask is missing
        if mask_a is None or mask_b is None:
            print("[warn] SAMLoss: missing mask (mask_a or mask_b is None); falling back to full L1.")
            return self.l1(img_a, img_b)

        # Invert each mask separately to get background regions
        background_a = ~mask_a  # Background of img_a
        background_b = ~mask_b  # Background of img_b
        
        # Check if backgrounds are empty
        if not background_a.any():
            print("[warn] SAMLoss: background_a empty; falling back to full L1.")
            return self.l1(img_a, img_b)
        if not background_b.any():
            print("[warn] SAMLoss: background_b empty; falling back to full L1.")
            return self.l1(img_a, img_b)

        # Convert to tensors and prepare for broadcasting
        background_mask_a = torch.from_numpy(background_a).to(img_a.device, dtype=img_a.dtype)
        background_mask_b = torch.from_numpy(background_b).to(img_b.device, dtype=img_b.dtype)
        
        # Add channel dimension for broadcasting: (H, W) -> (1, H, W)
        if background_mask_a.dim() == 2:
            background_mask_a = background_mask_a.unsqueeze(0)
        if background_mask_b.dim() == 2:
            background_mask_b = background_mask_b.unsqueeze(0)

        # Optionally save background-only visualizations for inspection
        if self._save_idx % self.save_every == 0:
            self._save_background_image(img_a, background_a, "img_a")
            self._save_background_image(img_b, background_b, "img_b")
        self._save_idx += 1

        # Compute L1 loss between the background regions of each image
        # Normalize by number of background pixels instead of mean over all pixels
        masked_img_a = img_a * background_mask_a
        masked_img_b = img_b * background_mask_b
        
        # Compute L1 difference
        l1_diff = torch.abs(masked_img_a - masked_img_b)
        
        # Count background pixels (use intersection of both background masks for normalization)
        # This ensures we normalize by pixels that are background in both images
        background_intersection = background_mask_a * background_mask_b
        num_background_pixels = background_intersection.sum()
        
        if num_background_pixels > 0:
            # Sum over all dimensions and normalize by number of background pixels
            loss = l1_diff.sum() / num_background_pixels
        else:
            # Fallback to mean if no background pixels
            loss = l1_diff.mean()
        
        return loss

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor, text="dog") -> torch.Tensor:
        """
        img_a, img_b: tensors shaped (C, H, W) or (B, C, H, W)
        text: Text to use for best_mask selection on img_b (default: "dog")
        """
        if img_a.dim() == 4:
            if img_a.shape[0] != img_b.shape[0]:
                raise ValueError("Batch size mismatch between img_a and img_b.")
            losses = [self._background_l1(a, b, text=text) for a, b in zip(img_a, img_b)]
            return torch.stack(losses).mean()

        return self._background_l1(img_a, img_b, text=text)
