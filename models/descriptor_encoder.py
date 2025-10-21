import torch
import torch.nn as nn
from typing import List
from ultralytics import YOLO
import cv2
import numpy as np

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
    def __init__(self, encoded_dim=128):
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
    def __init__(self, encoded_dim=128):
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
    
class Semantic_DescEncoder():
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encdec = EncoderDecoder(encoded_dim=128)
        self.encdec.load_state_dict(torch.load('./models/weights/semantic_encoder.pth'))
        self.encdec.to(self.device)

        self.semenc = Encoder(encoded_dim=128).to(self.device)
        self.semenc.encoder.load_state_dict(self.encdec.encoder.state_dict())

        self.desenc = DescriptorFeatureExtractor()
        self.desenc.load_state_dict(torch.load('./models/weights/orb_feature_extractor6.pth')) #See colab for ext. ID
        self.desenc.to(self.device)

        self.yolo = YOLO("./models/weights/yolo.pt")
        self.orb = cv2.ORB_create(1000)

    def encode(self, image,sem_background=None):
        #image = np.array(image.cpu())[0][0]
        image = image.astype('uint8')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)                
        kp, desc = self.orb.detectAndCompute(image, None)
        if desc is None:
            print("No keypoints detected by ORB detector.")
            return {
            'keypoints': [],
            'scores': [],
            'descriptors': [],
        }
        
        td = torch.tensor(desc, dtype=torch.float32)
        # If a single descriptor, add batch dimension
        if td.dim() == 1:
            td = td.unsqueeze(0)
        td = td.to(self.device)
        with torch.no_grad():  # Disable gradient calculation for inference
            enc_td = self.desenc(td)
        enc_td = enc_td.cpu().numpy()

        cv2.imshow("Semantic Background",sem_background)
        cv2.waitKey(1)
        sem_background = torch.tensor(sem_background, dtype=torch.float32)
        sem_background = sem_background.unsqueeze(0).unsqueeze(0).to(self.device)
 
        encoded = self.semenc(sem_background)
        bottleneck_vector = (encoded.cpu().numpy())

        keypoints = np.array([[p.pt[0], p.pt[1]] for p in kp], dtype=np.float32)
        keypoints = torch.tensor(keypoints, dtype=torch.float32).to(self.device)

        responses = [p.response for p in kp]
        max_response = max(responses)
        scores = np.array([response / max_response for response in responses], np.float32)
        scores = torch.tensor(scores, dtype=torch.float32).to(self.device)

        descriptors = []
        for point in enc_td:
            #descriptors.append(point)
            descriptors.append(np.concatenate((bottleneck_vector[0],point)))
        descriptors = np.array(descriptors,np.float32)
        descriptors = torch.tensor(descriptors, dtype=torch.float32).to(self.device)

        return {
            'keypoints': keypoints,
            'scores': scores,
            'descriptors': descriptors,
        }


'''
# Initialize the model
keypoint_type = 'orb'  # or 'sift', depending on your use case
model = DescriptorFeatureExtractor(keypoint_type)

# Load the saved model weights
model.load_state_dict(torch.load('orb_feature_extractor.pth'))

# Set the model to evaluation mode
model.eval()

# Example input descriptor (modify as per your actual input data)
input_descriptor = torch.tensor([
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 
    0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 
    1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 
    2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2
], dtype=torch.float32)

# If a single descriptor, add batch dimension
if input_descriptor.dim() == 1:
    input_descriptor = input_descriptor.unsqueeze(0)

# Ensure the model is on the same device as the input
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
input_descriptor = input_descriptor.to(device)

# Run inference
with torch.no_grad():  # Disable gradient calculation for inference
    output_feature = model(input_descriptor)

print(output_feature)
'''
