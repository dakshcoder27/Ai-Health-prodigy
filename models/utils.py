import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import gc
import json
import numpy as np
import cv2


class DynamicUNet(nn.Module):
    def __init__(self, filters, input_channels=1, output_channels=1):
        super(DynamicUNet, self).__init__()

        if len(filters) != 5:
            raise Exception(f"Filter list size {len(filters)}, expected 5!")

        padding = 1
        ks = 3
        # Encoding Part of Network
        #   Block 1
        self.conv1_1 = nn.Conv2d(input_channels, filters[0], kernel_size=ks, padding=padding)
        self.conv1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)
        self.maxpool1 = nn.MaxPool2d(2)
        #   Block 2
        self.conv2_1 = nn.Conv2d(filters[0], filters[1], kernel_size=ks, padding=padding)
        self.conv2_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.maxpool2 = nn.MaxPool2d(2)
        #   Block 3
        self.conv3_1 = nn.Conv2d(filters[1], filters[2], kernel_size=ks, padding=padding)
        self.conv3_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.maxpool3 = nn.MaxPool2d(2)
        #   Block 4
        self.conv4_1 = nn.Conv2d(filters[2], filters[3], kernel_size=ks, padding=padding)
        self.conv4_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.maxpool4 = nn.MaxPool2d(2)
        
        # Bottleneck Part of Network.
        self.conv5_1 = nn.Conv2d(filters[3], filters[4], kernel_size=ks, padding=padding)
        self.conv5_2 = nn.Conv2d(filters[4], filters[4], kernel_size=ks, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(filters[4], filters[3], 2, stride=2)

        # Decoding Part of Network
        #   Block 4
        self.conv6_1 = nn.Conv2d(filters[4], filters[3], kernel_size=ks, padding=padding)
        self.conv6_2 = nn.Conv2d(filters[3], filters[3], kernel_size=ks, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(filters[3], filters[2], 2, stride=2)
        #   Block 3
        self.conv7_1 = nn.Conv2d(filters[3], filters[2], kernel_size=ks, padding=padding)
        self.conv7_2 = nn.Conv2d(filters[2], filters[2], kernel_size=ks, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(filters[2], filters[1], 2, stride=2)
        #   Block 2
        self.conv8_1 = nn.Conv2d(filters[2], filters[1], kernel_size=ks, padding=padding)
        self.conv8_2 = nn.Conv2d(filters[1], filters[1], kernel_size=ks, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(filters[1], filters[0], 2, stride=2)
        #   Block 1
        self.conv9_1 = nn.Conv2d(filters[1], filters[0], kernel_size=ks, padding=padding)
        self.conv9_2 = nn.Conv2d(filters[0], filters[0], kernel_size=ks, padding=padding)

        # Output Part of Network.
        self.conv10 = nn.Conv2d(filters[0], output_channels, kernel_size=ks, padding=padding)

    def forward(self, x):
        # Encoding Part of Network
        #   Block 1
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        pool1 = self.maxpool1(conv1)
        #   Block 2
        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        pool2 = self.maxpool2(conv2)
        #   Block 3
        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        pool3 = self.maxpool3(conv3)
        #   Block 4
        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        pool4 = self.maxpool4(conv4)

        # Bottleneck Part of Network.
        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))

        # Decoding Part of Network.
        #   Block 4
        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))
        
        #   Block 3
        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))
        #   Block 2
        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))
        
        #   Block 1
        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))

        # Output Part of Network
        output = F.sigmoid(self.conv10(conv9))         
        return output
    

class BrainTumorClassifier():
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def restore_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location='cpu'))

    def predict(self, data, threshold=0.5):
        self.model.eval()
        image = data['image']

        # Resize safely using OpenCV
        if image.shape != (512, 512):
            image = cv2.resize(image, (512, 512))

        # Normalize and convert to tensor
        image_tensor = torch.tensor(image, dtype=torch.float32) / 255.0  # Normalize to [0,1]
        image_tensor = image_tensor.unsqueeze(0).unsqueeze(0).to(self.device)  # Shape: [1, 1, 512, 512]

        with torch.no_grad():
            output = self.model(image_tensor).detach().cpu()
            output = (output > threshold).numpy().astype(np.uint8)

        # Remove batch and channel dimension: [1, 1, 512, 512] → [512, 512]
        return output[0, 0]



class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)