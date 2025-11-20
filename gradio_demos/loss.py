import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor

try:
    import lpips as lpips_lib
except ImportError:
    lpips_lib = None

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

class LPIPSLoss(torch.nn.Module):
    def __init__(self, net_type: str = "vgg"):
        super().__init__()
        if lpips_lib is None:
            raise ImportError("lpips package is required for LPIPSLoss (pip install lpips)")
        self.metric = lpips_lib.LPIPS(net=net_type)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if x.min() >= 0 and x.max() <= 1:
            x = x * 2 - 1
        if y.min() >= 0 and y.max() <= 1:
            y = y * 2 - 1
        return self.metric(x, y).mean()

