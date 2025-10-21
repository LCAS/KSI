import torch
import torch.nn as nn
from typing import List
from ultralytics import YOLO
import cv2
import numpy as np

ENCDIM = 256

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

def MLP(channels: List[int], do_bn: bool = True) -> nn.Module:
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(
            nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)

class DescriptorFeatureExtractor(nn.Module):
    def __init__(self, keypoint_type='orb'):
        super(DescriptorFeatureExtractor, self).__init__()
        self.keypoint_type = keypoint_type

        if keypoint_type == 'sift':
            input_dim = 128  # assuming the input is already in the form of a 128-dimensional vector
        elif keypoint_type == 'orb':
            input_dim = 32  # ORB typically produces 32-dimensional descriptors
        else:
            raise ValueError("Invalid keypoint type. Choose from ['sift', 'orb']")

        self.mlp = MLP([input_dim, 1024, 512, 256]) #MLP([input_dim, 64, 128, 256, 512, 256, 128])

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(2)
        elif x.dim() != 3:
            raise ValueError(f"Expected 2D or 3D input tensor, got {x.dim()}D tensor")

        for layer in self.mlp:
            if isinstance(layer, nn.BatchNorm1d) and x.size(0) == 1:
                # Skip batch normalization if batch size is 1
                continue
            x = layer(x)

        x = x.squeeze(2)
        return x

class EncoderDecoder(nn.Module):
    def __init__(self, encoded_dim=ENCDIM):
        super(EncoderDecoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, encoded_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(encoded_dim, 128 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (128, 16, 16)),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# Define the Encoder-Only Network
class Encoder(nn.Module):
    def __init__(self, encoded_dim=ENCDIM):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, encoded_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        return encoded

class ORB(nn.Module):
    def __init__(self):
        super(ORB, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.desenc = DescriptorFeatureExtractor()
        self.desenc.load_state_dict(torch.load('./models/weights/orb_feature_extractor6.pth')) #See colab for ext. ID
        self.desenc.to(self.device)

        self.orb = cv2.ORB_create(2500)

        self.encdec = EncoderDecoder()
        self.encdec.load_state_dict(torch.load(f'./models/weights/{ENCDIM}_BN.pth'))
        self.encdec.to(self.device)

        self.semenc = Encoder().to(self.device)
        self.semenc.encoder.load_state_dict(self.encdec.encoder.state_dict())

        print('Loaded Semantic Encoder model')

    def forward(self, data, masks):
        image = data['image'].astype('uint8')#.detach().cpu().numpy().astype('uint8')[0,0]*255.0

        # Extract Keypoints and Descriptors
        kp, desc = self.orb.detectAndCompute(image, None)

        if desc is None:
            print("No keypoints detected by ORB detector.")
            return {
            'keypoints': [],
            'scores': [],
            'descriptors': [],
            'indexes' : [],
        }

        keypoints = np.array([[p.pt[0], p.pt[1]] for p in kp], dtype=np.int32)
        
        td = torch.tensor(desc, dtype=torch.float32)
        # If a single descriptor, add batch dimension
        if td.dim() == 1:
            td = td.unsqueeze(0)
        td = td.to(self.device)
        with torch.no_grad():  # Disable gradient calculation for inference
            descriptors = self.desenc(td)
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)

        # Extract Scores
        responses = [p.response for p in kp]
        max_response = max(responses)
        scores = np.array([response / max_response for response in responses], np.float32)
        scores = [torch.tensor(scores, dtype=torch.float32).to(self.device)]

        if masks is not None:
            mask_indexes = find_nearest_masks_for_keypoints(masks, keypoints)

            semantic_descriptors = []
            for idx, desc in enumerate(descriptors):
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
                mask_indexes = np.full((len(keypoints)), -1, dtype=np.int64)
                semantic_descriptors = descriptors.squeeze(0).cpu()

        keypoints = [torch.tensor(keypoints, dtype=torch.float32).to(self.device)]
        descriptors = torch.tensor(semantic_descriptors, dtype=torch.float32).to(self.device).T.unsqueeze(0) #for unbranched
        descriptors = torch.nn.functional.normalize(descriptors, p=2, dim=1)
        mask_indexes = torch.tensor(mask_indexes, dtype=torch.int64).unsqueeze(0)

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
            'indexes' : mask_indexes,
        }