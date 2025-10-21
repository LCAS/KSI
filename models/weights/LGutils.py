import collections.abc as collections
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional, Tuple, Union

import cv2
import kornia
import numpy as np
import torch

def find_nearest_masks_for_keypoints(masks, keypoints):
    N = masks.shape[0]
    result_indices = []
    points_with_255 = []
    
    for i in range(N):
            # Get coordinates of all points with value 255 in the current mask
            points_with_255.append(np.argwhere(masks[i] == 255))

    for keypoint in keypoints:
        x, y = keypoint
        min_distance = float('inf')
        nearest_mask_index = -1
        
        for i in range(N):            
            if points_with_255[i].size == 0:
                # If there are no points with 255 in this mask, skip it
                continue
            
            # Calculate the squared Euclidean distance to the keypoint for each point with value 255
            distances = np.sqrt((points_with_255[i][:, 0] - y) ** 2 + (points_with_255[i][:, 1] - x) ** 2)
            
            # Find the minimum distance in this mask
            min_dist_in_mask = np.min(distances)
            
            # Update the nearest mask and distance if the current one is closer
            if min_dist_in_mask < min_distance:
                min_distance = min_dist_in_mask
                nearest_mask_index = i
        
        # Store the nearest mask index for the current keypoint
        if min_distance < 2.0:
            result_indices.append(nearest_mask_index)
        else:
            result_indices.append(-1)
    
    return np.array(result_indices)

class ImagePreprocessor:
    default_conf = {
        "resize": None,  # target edge length, None for no resizing
        "side": "long",
        "interpolation": "bilinear",
        "align_corners": None,
        "antialias": True,
    }

    def __init__(self, **conf) -> None:
        super().__init__()
        self.conf = {**self.default_conf, **conf}
        self.conf = SimpleNamespace(**self.conf)

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize and preprocess an image, return image and resize scale"""
        h, w = img.shape[-2:]
        if self.conf.resize is not None:
            img = kornia.geometry.transform.resize(
                img,
                self.conf.resize,
                side=self.conf.side,
                antialias=self.conf.antialias,
                align_corners=self.conf.align_corners,
            )
        scale = torch.Tensor([img.shape[-1] / w, img.shape[-2] / h]).to(img)
        return img, scale


def map_tensor(input_, func: Callable):
    string_classes = (str, bytes)
    if isinstance(input_, string_classes):
        return input_
    elif isinstance(input_, collections.Mapping):
        return {k: map_tensor(sample, func) for k, sample in input_.items()}
    elif isinstance(input_, collections.Sequence):
        return [map_tensor(sample, func) for sample in input_]
    elif isinstance(input_, torch.Tensor):
        return func(input_)
    else:
        return input_


def batch_to_device(batch: dict, device: str = "cpu", non_blocking: bool = True):
    """Move batch (dict) to device"""

    def _func(tensor):
        return tensor.to(device=device, non_blocking=non_blocking).detach()

    return map_tensor(batch, _func)


def rbd(data: dict) -> dict:
    """Remove batch dimension from elements in data"""
    return {
        k: v[0] if isinstance(v, (torch.Tensor, np.ndarray, list)) else v
        for k, v in data.items()
    }


def read_image(path: Path, grayscale: bool = False) -> np.ndarray:
    """Read an image from path as RGB or grayscale"""
    if not Path(path).exists():
        raise FileNotFoundError(f"No image at path {path}.")
    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(path), mode)
    if image is None:
        raise IOError(f"Could not read image at {path}.")
    if not grayscale:
        image = image[..., ::-1]
    return image


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def resize_image(
    image: np.ndarray,
    size: Union[List[int], int],
    fn: str = "max",
    interp: Optional[str] = "area",
) -> np.ndarray:
    """Resize an image to a fixed size, or according to max or min edge."""
    h, w = image.shape[:2]

    fn = {"max": max, "min": min}[fn]
    if isinstance(size, int):
        scale = size / fn(h, w)
        h_new, w_new = int(round(h * scale)), int(round(w * scale))
        scale = (w_new / w, h_new / h)
    elif isinstance(size, (tuple, list)):
        h_new, w_new = size
        scale = (w_new / w, h_new / h)
    else:
        raise ValueError(f"Incorrect new size: {size}")
    mode = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
        "nearest": cv2.INTER_NEAREST,
        "area": cv2.INTER_AREA,
    }[interp]
    return cv2.resize(image, (w_new, h_new), interpolation=mode), scale


def load_image_LG(path: Path, resize: int = None, **kwargs) -> torch.Tensor:
    image = read_image(path)
    if resize is not None:
        image, _ = resize_image(image, resize, **kwargs)
    return numpy_image_to_torch(image)


class Extractor(torch.nn.Module):
    def __init__(self, **conf):
        super().__init__()
        self.conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    @torch.no_grad()
    def extract(self, img: torch.Tensor, masks, **conf) -> dict:
        """Perform extraction with online resizing"""
        if img.dim() == 3:
            img = img[None]  # add batch dim
        assert img.dim() == 4 and img.shape[0] == 1
        shape = img.shape[-2:][::-1]
        img, scales = ImagePreprocessor(**{**self.preprocess_conf, **conf})(img)
        print(img.shape)
        feats = self.forward({"image": img})
        feats["image_size"] = torch.tensor(shape)[None].to(img).float()
        feats["keypoints"] = (feats["keypoints"] + 0.5) / scales[None] - 0.5
        print(np.max(feats["keypoints"][0].cpu().numpy()[:,1]),np.max(feats["keypoints"][0].cpu().numpy()[:,1]))
        if masks is not None:
            mask_indexes = find_nearest_masks_for_keypoints(masks, feats["keypoints"][0].cpu().numpy())
            semantic_descriptors = []
            for idx, desc in enumerate(feats["descriptors"]):
                if mask_indexes[idx]>= 0:
                    sem_background = masks[mask_indexes[idx]]
                    sem_background = torch.tensor(sem_background, dtype=torch.float32).to(self.device)
                    sem_background = torch.nn.functional.interpolate(
                                        sem_background.unsqueeze(0).unsqueeze(0),
                                        size=(128, 128),  # Adjust to expected input size
                                        mode='bilinear',
                                        align_corners=False
                                    )
                    encoded = self.semenc(sem_background)
                    #encoded = torch.nn.functional.normalize(encoded, p=2, dim=0)
                    bottleneck_vector = (encoded.cpu().numpy())
                    #semantic_descriptors.append(np.concatenate((bottleneck_vector[0],reduced_desc)))
                    semantic_descriptors.append(desc.cpu().numpy()+bottleneck_vector[0])
                    
                else:
                    semantic_descriptors.append(desc.cpu().numpy())
        else:
                mask_indexes = np.full((len(feats["keypoints"][0])), -1, dtype=np.int64)
                semantic_descriptors = feats["descriptors"].squeeze(0).cpu()

        descriptors = torch.tensor(semantic_descriptors, dtype=torch.float32).to(self.device).T.unsqueeze(0) #for unbranched
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        feats["indexes"] = torch.tensor(mask_indexes, dtype=torch.int64).unsqueeze(0)

        return feats


def match_pair(
    extractor,
    matcher,
    image0: torch.Tensor,
    image1: torch.Tensor,
    device: str = "cpu",
    **preprocess,
):
    """Match a pair of images (image0, image1) with an extractor and matcher"""
    feats0 = extractor.extract(image0, **preprocess)
    feats1 = extractor.extract(image1, **preprocess)
    matches01 = matcher({"image0": feats0, "image1": feats1})
    data = [feats0, feats1, matches01]
    # remove batch dim and move to target device
    feats0, feats1, matches01 = [batch_to_device(rbd(x), device) for x in data]
    return feats0, feats1, matches01
