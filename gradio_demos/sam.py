import os
import urllib
from typing import Any, Callable, Dict, List, Tuple, Union

import clip
import cv2
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

try:
    import kornia.augmentation as K
    KORNIA_AVAILABLE = True
except ImportError:
    KORNIA_AVAILABLE = False
    print("Warning: kornia not available. Install with: pip install kornia")

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False
    print("Warning: lpips not available. Install with: pip install lpips")


class Losses:
   
    @staticmethod
    def d_clip_loss(x, y, use_cosine=False):
        x = F.normalize(x, dim=-1)
        y = F.normalize(y, dim=-1)
        
        if use_cosine:
            distance = 1 - (x @ y.t()).squeeze()
        else:
            distance = (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)
        
        return distance
    
    @staticmethod
    def range_loss(input):
        return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


class ImageAugmentations(nn.Module):
    def __init__(self, output_size, augmentations_number, p=0.7):
        super().__init__()
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia is required for ImageAugmentations. Install with: pip install kornia")
        self.output_size = output_size
        self.augmentations_number = augmentations_number
        self.augmentations = nn.Sequential(
            K.RandomAffine(degrees=15, translate=0.1, p=p, padding_mode="border"),  # type: ignore
            K.RandomPerspective(0.7, p=p),
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((self.output_size, self.output_size))

    def forward(self, input):
        resized_images = self.avg_pool(input)
        batch_size = input.shape[0]
        # Tile to get batch_size * (1 + augmentations_number) images total
        # First batch_size are non-augmented, rest are augmented
        resized_images = torch.tile(resized_images, dims=(1 + self.augmentations_number, 1, 1, 1))
        non_augmented_batch = resized_images[:batch_size]
        augmented_batch = self.augmentations(resized_images[batch_size:])
        updated_batch = torch.cat([non_augmented_batch, augmented_batch], dim=0)
        return updated_batch


class Segmentation:    
    def __init__(
        self,
        checkpoint_path: str = None,
        checkpoint_name: str = "sam_vit_h_4b8939.pth",
        checkpoint_url: str = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        model_type: str = "default",
        clip_model_name: str = "ViT-B/32",
        max_width: int = 1024,
        max_height: int = 1024,
        top_k_objects: int = 100,
        device: str = None,
    ):
        self.checkpoint_path = checkpoint_path or os.path.join(
            os.path.expanduser("~"), ".cache", "SAM"
        )
        self.checkpoint_name = checkpoint_name
        self.checkpoint_url = checkpoint_url
        self.model_type = model_type
        self.clip_model_name = clip_model_name
        self.max_width = max_width
        self.max_height = max_height
        self.top_k_objects = top_k_objects
        
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        
        self._mask_generator = None
        self._clip_model = None
        self._clip_preprocess = None
        
        self.image_augmentations = None
        self._aug_output_size = 224
        self._aug_number = 1
        self._aug_p = 0.7

        self.clip_normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
        )
        
        self.lpips_model = None
        self.init_image = None
    
    def set_init_image(self, init_image_tensor: torch.Tensor):
        if init_image_tensor.dim() == 3:
            init_image_tensor = init_image_tensor.unsqueeze(0)
        
        init_image_tensor = init_image_tensor.detach().to(self.device)
        
        if init_image_tensor.min() >= 0 and init_image_tensor.max() <= 1:
            init_image_tensor = init_image_tensor.mul(2).sub(1)
        
        self.init_image = init_image_tensor
        
    def _load_lpips_model(self):
        if not LPIPS_AVAILABLE:
            raise ImportError("lpips is required. Install with: pip install lpips")
        
        if self.lpips_model is None:
            self.lpips_model = lpips.LPIPS(net="vgg").to(self.device)
            self.lpips_model.eval()
            for param in self.lpips_model.parameters():
                param.requires_grad = False
        return self.lpips_model
    
    def _load_mask_generator(self) -> SamAutomaticMaskGenerator:
        if self._mask_generator is None:
            if not os.path.exists(self.checkpoint_path):
                os.makedirs(self.checkpoint_path)
            checkpoint = os.path.join(self.checkpoint_path, self.checkpoint_name)
            if not os.path.exists(checkpoint):
                urllib.request.urlretrieve(self.checkpoint_url, checkpoint)
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint).to(
                self.device
            )
            sam.eval()
            for param in sam.parameters():
                param.requires_grad = False
            self._mask_generator = SamAutomaticMaskGenerator(sam)
        return self._mask_generator
    
    def _load_clip(
        self,
    ) -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
        if self._clip_model is None or self._clip_preprocess is None:
            self._clip_model, self._clip_preprocess = clip.load(
                self.clip_model_name, device=self.device
            )
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            for param in self._clip_model.parameters():
                param.requires_grad = False
        return self._clip_model, self._clip_preprocess
    
    def _load_image_augmentations(self, output_size: int = None, augmentations_number: int = None, p: float = None) -> ImageAugmentations:
        if not KORNIA_AVAILABLE:
            raise ImportError("kornia is required for image augmentations. Install with: pip install kornia")
        
        if self.image_augmentations is None:
            output_size = output_size if output_size is not None else self._aug_output_size
            augmentations_number = augmentations_number if augmentations_number is not None else self._aug_number
            p = p if p is not None else self._aug_p
            
            self.image_augmentations = ImageAugmentations(
                output_size=output_size,
                augmentations_number=augmentations_number,
                p=p
            ).to(self.device)
            self.image_augmentations.eval()
            for param in self.image_augmentations.parameters():
                param.requires_grad = False
        return self.image_augmentations
    
    def _adjust_image_size(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if height > width:
            if height > self.max_height:
                height, width = self.max_height, int(
                    self.max_height / height * width
                )
        else:
            if width > self.max_width:
                height, width = int(self.max_width / width * height), self.max_width
        image = cv2.resize(image, (width, height))
        return image
    
    @torch.no_grad()
    def _get_score(self, crop: PIL.Image.Image, texts: List[str]) -> float:
        model, preprocess = self._load_clip()
        preprocessed = preprocess(crop).unsqueeze(0).to(self.device)
        tokens = clip.tokenize(texts).to(self.device)
        logits_per_image, _ = model(preprocessed, tokens)
        similarity = logits_per_image.softmax(-1).cpu()
        return float(similarity[0, 0])
    
    def _crop_image(
        self, image: np.ndarray, mask: Dict[str, Any]
    ) -> PIL.Image.Image:
        x, y, w, h = mask["bbox"]
        masked = image * np.expand_dims(mask["segmentation"], -1)
        crop = masked[y : y + h, x : x + w]
        
        if h > w:
            top, bottom, left, right = 0, 0, (h - w) // 2, (h - w) // 2
        else:
            top, bottom, left, right = (w - h) // 2, (w - h) // 2, 0, 0
        
        crop = cv2.copyMakeBorder(
            crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
        crop = PIL.Image.fromarray(crop)
        return crop
    
    def _get_texts(self, query: str) -> List[str]:
        return [f"a picture of {query}", "a picture of background"]
    
    def _filter_masks(
        self,
        image: np.ndarray,
        masks: List[Dict[str, Any]],
        predicted_iou_threshold: float,
        stability_score_threshold: float,
        query: str,
        clip_threshold: float,
    ) -> List[Dict[str, Any]]:
        filtered_masks: List[Dict[str, Any]] = []
        
        for mask in sorted(masks, key=lambda mask: mask["area"])[
            -self.top_k_objects :
        ]:
            if (
                mask["predicted_iou"] < predicted_iou_threshold
                or mask["stability_score"] < stability_score_threshold
                or image.shape[:2] != mask["segmentation"].shape[:2]
            ):
                continue
            
            if query:
                score = self._get_score(
                    self._crop_image(image, mask), self._get_texts(query)
                )
                if score < clip_threshold:
                    continue
            
            filtered_masks.append(mask)
        
        return filtered_masks
    
    def calculate_mask(
        self,
        image: Union[str, np.ndarray, PIL.Image.Image],
        text: str,
        predicted_iou_threshold: float = 0.9,
        stability_score_threshold: float = 0.8,
        clip_threshold: float = 0.85,
    ) -> Dict[str, Any]:
        if isinstance(image, str):
            image = cv2.imread(image, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif isinstance(image, PIL.Image.Image):
            image = np.array(image)
            if len(image.shape) == 3 and image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                pass
            else:
                raise ValueError(
                    "Unsupported image format. Expected RGB image with 3 channels."
                )
        else:
            raise TypeError(
                f"Unsupported image type: {type(image)}. "
                "Expected str (file path), numpy.ndarray, or PIL.Image.Image"
            )
        
        image = self._adjust_image_size(image)
        
        mask_generator = self._load_mask_generator()
        masks = mask_generator.generate(image)
        
        filtered_masks = self._filter_masks(
            image,
            masks,
            predicted_iou_threshold,
            stability_score_threshold,
            text,
            clip_threshold,
        )
        
        height, width = image.shape[:2]
        foreground_mask = np.zeros((height, width), dtype=bool)
        
        if len(filtered_masks) > 0:
            for mask_dict in filtered_masks:
                mask_seg = mask_dict["segmentation"]
                if mask_seg.shape == (height, width):
                    foreground_mask = foreground_mask | mask_seg
        
        background_mask = ~foreground_mask
        
        return {
            "foreground_mask": foreground_mask,
            "background_mask": background_mask,
            "filtered_masks": filtered_masks
        }

    def calculate_total_loss(
        self,
        image: Union[str, np.ndarray, PIL.Image.Image, torch.Tensor],
        query_text: str,
        target_text: str,
        predicted_iou_threshold: float = 0.9,
        stability_score_threshold: float = 0.8,
        clip_threshold: float = 0.85,
        clip_guidance_lambda: float = 1000,
        range_lambda: float = 50,
        lpips_sim_lambda: float = 1000,
        l2_sim_lambda: float = 10000,
    ) -> torch.Tensor:
        # Handle tensor input (preserve gradients)
        image_tensor = None
        if isinstance(image, torch.Tensor):
            # If tensor is provided, use it directly for loss computation
            if image.dim() == 4:  # [B, C, H, W]
                image_tensor = image.to(self.device)
            elif image.dim() == 3:  # [C, H, W]
                image_tensor = image.unsqueeze(0).to(self.device)
            else:
                raise ValueError(f"Unsupported tensor shape: {image.shape}")
            
            # Normalize tensor if needed (assuming it's in [0, 1] range)
            if image_tensor.min() >= 0 and image_tensor.max() <= 1:
                image_tensor = image_tensor.mul(2).sub(1)
            
            # Convert to numpy/PIL for mask calculation (detached copy)
            image_np = image_tensor[0].detach().cpu().permute(1, 2, 0).numpy()
            image_np = ((image_np + 1) / 2).clip(0, 1)
            image_np = (image_np * 255).astype(np.uint8)
            image_for_mask = PIL.Image.fromarray(image_np)
        else:
            image_for_mask = image
        
        result = self.calculate_mask(
            image=image_for_mask,
            text=query_text,
            predicted_iou_threshold=predicted_iou_threshold,
            stability_score_threshold=stability_score_threshold,
            clip_threshold=clip_threshold,
        )
        
        foreground_mask = result["foreground_mask"]
        
        # Use the tensor directly if provided, otherwise convert from image
        if image_tensor is not None:
            x_tensor = image_tensor
            # x_tensor should inherit requires_grad from image_tensor
            # For init_image, use a detached copy of the tensor
            if self.init_image is None:
                self.init_image = x_tensor.detach().clone()
        else:
            if isinstance(image, str):
                x_in = cv2.imread(image, cv2.IMREAD_COLOR)
                x_in = cv2.cvtColor(x_in, cv2.COLOR_BGR2RGB)
            elif isinstance(image, PIL.Image.Image):
                x_in = np.array(image)
                if len(x_in.shape) == 3 and x_in.shape[2] == 4:
                    x_in = cv2.cvtColor(x_in, cv2.COLOR_RGBA2RGB)
            elif isinstance(image, np.ndarray):
                x_in = image.copy()
                if len(x_in.shape) == 3 and x_in.shape[2] == 3:
                    pass
                else:
                    raise ValueError(
                        "Unsupported image format. Expected RGB image with 3 channels."
                    )
            else:
                raise TypeError(
                    f"Unsupported image type: {type(image)}. "
                    "Expected str (file path), numpy.ndarray, PIL.Image.Image, or torch.Tensor"
                )
            
            x_in = self._adjust_image_size(x_in)
            
            x_tensor = torch.from_numpy(x_in).permute(2, 0, 1).float()
            if x_tensor.max() > 1.0:
                x_tensor = x_tensor / 255.0
            x_tensor = x_tensor.unsqueeze(0).to(self.device)
            x_tensor = x_tensor.mul(2).sub(1)
            
            if self.init_image is None:
                init_image_tensor = torch.from_numpy(x_in).permute(2, 0, 1).float()
                if init_image_tensor.max() > 1.0:
                    init_image_tensor = init_image_tensor / 255.0
                init_image_tensor = init_image_tensor.unsqueeze(0).to(self.device)
                self.init_image = init_image_tensor.mul(2).sub(1).detach()
        
        foreground_mask_tensor = torch.from_numpy(foreground_mask).float().to(self.device)
        background_mask = result["background_mask"]
        background_mask_tensor = torch.from_numpy(background_mask).float().to(self.device)
        
        mask_expanded = foreground_mask_tensor.unsqueeze(0).unsqueeze(0)
        masked_input_tensor = x_tensor * mask_expanded
        
        # Verify x_tensor has gradients (critical for gradient flow)
        if not x_tensor.requires_grad:
            raise RuntimeError(
                f"x_tensor does not require gradients. "
                f"This breaks the computation graph. "
                f"Input image type: {type(image)}, "
                f"image_tensor.requires_grad: {image_tensor.requires_grad if image_tensor is not None else 'N/A'}"
            )
        
        image_aug = self._load_image_augmentations()
        augmented_input = image_aug(masked_input_tensor)
        
        clip_in = self.clip_normalize(augmented_input)
        
        clip_model, _ = self._load_clip()
        # Ensure gradients flow through CLIP model
        # Even though CLIP params don't require grad, we need gradients through the forward pass
        # CLIP model in eval mode should still preserve gradients through forward pass
        image_embeds = clip_model.encode_image(clip_in).float()
        text_tokens = clip.tokenize([target_text]).to(self.device)
        text_embeds = clip_model.encode_text(text_tokens).float()
        
        dist = Losses.d_clip_loss(image_embeds, text_embeds, use_cosine=False)
        
        batch_size = masked_input_tensor.shape[0]
        # Initialize with a value that has gradients
        if batch_size > 0:
            clip_loss = dist[0 :: batch_size].mean()
            for i in range(1, batch_size):
                clip_loss = clip_loss + dist[i :: batch_size].mean()
        else:
            # Fallback: create a tensor that requires grad from x_tensor
            # This should not happen in practice, but handle edge case
            clip_loss = (x_tensor * 0.0).sum() if x_tensor.requires_grad else torch.tensor(0.0, device=self.device, requires_grad=True)
        
        background_mask_expanded = background_mask_tensor.unsqueeze(0).unsqueeze(0)
        masked_bg_tensor = x_tensor * background_mask_expanded
        
        range_loss_value = Losses.range_loss(x_tensor).sum()
        
        lpips_model = self._load_lpips_model()
        lpips_loss = lpips_model(masked_bg_tensor, self.init_image).sum()
        
        l2_loss = F.mse_loss(masked_bg_tensor, self.init_image)
        
        # Initialize loss with a value that has gradients
        # Use clip_loss as base (it always has gradients from x_tensor)
        # Even if clip_guidance_lambda is 0, we multiply by 0.0 to preserve gradient graph
        loss = clip_loss * clip_guidance_lambda
        
        if range_lambda != 0:
            loss = loss + range_loss_value * range_lambda
        
        if lpips_sim_lambda != 0:
            loss = loss + lpips_loss * lpips_sim_lambda
        
        if l2_sim_lambda != 0:
            loss = loss + l2_loss * l2_sim_lambda
        
        return loss
