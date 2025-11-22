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

    def _largest_mask(self, img: torch.Tensor):
        """
        Generate the largest SAM mask for a single image tensor shaped (C, H, W).
        Returns a numpy boolean array of shape (H, W) or None if no masks are found.
        """
        img_np = img.detach().permute(1, 2, 0).cpu().numpy()
        masks = self.mask_generator.generate(img_np)
        if len(masks) == 0:
            return None
        largest = max(masks, key=lambda m: m.get("area", 0))
        return largest["segmentation"].astype(bool)

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

    def _background_l1(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        mask_a = self._largest_mask(img_a)
        mask_b = self._largest_mask(img_b)

        # Fallback to full L1 if either mask is missing
        if mask_a is None or mask_b is None:
            print("[warn] SAMLoss: missing mask (mask_a or mask_b is None); falling back to full L1.")
            return self.l1(img_a, img_b)

        background = ~(mask_a | mask_b)
        if not background.any():
            print("[warn] SAMLoss: background empty after combining masks; falling back to full L1.")
            return self.l1(img_a, img_b)

        background_mask = torch.from_numpy(background).to(img_a.device, dtype=img_a.dtype)
        background_mask = background_mask.unsqueeze(0)  # broadcast over channels

        # Optionally save background-only visualizations for inspection
        if self._save_idx % self.save_every == 0:
            self._save_background_image(img_a, background, "img_a")
            self._save_background_image(img_b, background, "img_b")
        self._save_idx += 1

        return self.l1(img_a * background_mask, img_b * background_mask)

    def forward(self, img_a: torch.Tensor, img_b: torch.Tensor) -> torch.Tensor:
        """
        img_a, img_b: tensors shaped (C, H, W) or (B, C, H, W)
        """
        if img_a.dim() == 4:
            if img_a.shape[0] != img_b.shape[0]:
                raise ValueError("Batch size mismatch between img_a and img_b.")
            losses = [self._background_l1(a, b) for a, b in zip(img_a, img_b)]
            return torch.stack(losses).mean()

        return self._background_l1(img_a, img_b)
