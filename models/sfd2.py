import torch
import numpy as np
from pathlib import Path
from models.sfd2_nets.sfd2 import ResSegNetV2
from models.sfd2_nets.extractor import extract_resnet_return
from .ksi_encoders import EncoderDecoder, Encoder

class SFD2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ENCDIM = 128
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        model_path = Path(__file__).parent / 'weights/sfd2.pth'
        self.model = ResSegNetV2(outdim=128, require_stability=self.config["model"]["use_stability"]).eval()
        checkpoint = torch.load(model_path, weights_only=False, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'], strict=False)
        self.model.to(self.device)

        self.extractor = extract_resnet_return
        print("Loaded SFD2 model.")

        self.encdec = EncoderDecoder(self.ENCDIM)
        self.encdec.load_state_dict(torch.load(f'./models/weights/{self.ENCDIM}_BN.pth'))
        self.encdec.to(self.device)

        self.semenc = Encoder(self.ENCDIM).to(self.device)
        self.semenc.encoder.load_state_dict(self.encdec.encoder.state_dict())

        print('Loaded Semantic Encoder model')

    def find_nearest_masks_for_keypoints(self, masks, keypoints, threshold=2.0):
        """ masks: [N, H, W] numpy arrays; keypoints: [N, 2] torch tensor """
        N = len(masks)
        result_indices = []
        points_with_255 = [np.argwhere(m == 255) for m in masks]

        keypoints_np = keypoints.cpu().numpy()
        for keypoint in keypoints_np:
            y, x = keypoint
            min_distance = float('inf')
            nearest_mask_index = -1

            for i in range(N):
                if points_with_255[i].size == 0:
                    continue
                distances = np.sqrt((points_with_255[i][:, 0] - y) ** 2 + (points_with_255[i][:, 1] - x) ** 2)
                min_dist_in_mask = np.min(distances)
                if min_dist_in_mask < min_distance:
                    min_distance = min_dist_in_mask
                    nearest_mask_index = i

            if min_distance < threshold:
                result_indices.append(nearest_mask_index)
            else:
                result_indices.append(-1)

        return torch.tensor(result_indices, dtype=torch.int64, device=self.device)

    @torch.no_grad()
    def forward(self, data, masks=None):
        """ data: dict with key 'image' (Tensor of shape [1, 3, H, W]) """
        img = data["image"].unsqueeze(0).to(self.device)

        pred = self.extractor(
            self.model,
            img=img,
            topK=self.config["model"]["max_keypoints"],
            mask=None,
            conf_th=self.config["model"]["conf_th"],
            scales=self.config["model"]["scales"],
        )
        
        # Convert descriptors to match expected shape: [B, C, N]
        descriptors = pred["descriptors"].transpose(1, 0)
        descriptors = torch.tensor(descriptors, dtype=torch.float32, device=self.device)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        descriptors = descriptors.unsqueeze(0)  # [1, C, N]

        # Format keypoints (H, W -> X, Y) and score list
        keypoints = pred["keypoints"].transpose(1, 0)
        keypoints = torch.tensor(pred["keypoints"], dtype=torch.float32)
        keypoints = torch.flip(keypoints, dims=[1]).unsqueeze(0)  # [1, N, 2]
        scores = torch.tensor(pred["scores"], dtype=torch.float32).unsqueeze(0)  # [1, N]
        
        if self.config["ksi"]:
            if masks is not None:
                mask_indexes = self.find_nearest_masks_for_keypoints(masks, keypoints[0])

                bottleneck_vectors = []
                for mask in masks:
                    sem_background = torch.tensor(mask, dtype=torch.float32).to(self.device)
                    sem_background = torch.nn.functional.interpolate(
                                        sem_background.unsqueeze(0).unsqueeze(0),
                                        size=(128, 128),
                                        mode='bilinear',
                                        align_corners=False
                                    )
                    encoded = self.semenc(sem_background)
                    #encoded = torch.nn.functional.normalize(encoded, p=2, dim=0)
                    bottleneck_vector = (encoded.cpu().numpy())
                    bottleneck_vectors.append(bottleneck_vector[0])

                semantic_descriptors = []
                for idx, desc in enumerate(descriptors[0].T):
                    if mask_indexes[idx]>= 0:
                        semantic_descriptors.append(desc.cpu().numpy()+bottleneck_vectors[mask_indexes[idx]])
                    else:
                        semantic_descriptors.append(desc.cpu().numpy())
            else:
                semantic_descriptors = descriptors[0].T.cpu().numpy()
                mask_indexes = torch.full((keypoints.shape[1],), -1, dtype=torch.int64, device=self.device)
        else:
            semantic_descriptors = descriptors[0].T.cpu().numpy()
            if masks is not None:
                mask_indexes = self.find_nearest_masks_for_keypoints(masks, keypoints[0])
            else:
                mask_indexes = torch.full((keypoints.shape[1],), -1, dtype=torch.int64, device=self.device)
        descriptors = np.array(semantic_descriptors,np.float32)
        descriptors = torch.tensor(descriptors, dtype=torch.float32).to(self.device).T.unsqueeze(0)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        return {
            "keypoints": keypoints.to(self.device),
            "scores": scores.to(self.device),
            "descriptors": descriptors,  # already on self.device
            "indexes": mask_indexes.unsqueeze(0),  # [1, N]
        }
