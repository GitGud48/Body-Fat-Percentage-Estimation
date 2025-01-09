import torch
from torch import nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import requests
import zipfile
from tqdm.auto import tqdm

size=40
device ="cpu"
class_names = ['5-9%','10-14%','15-19%']

class TinyVGG(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*size*size,
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

model_0 = TinyVGG(input_shape=1, # number of color channels (3 for RGB) 
                  hidden_units=3, 
                  output_shape=len(class_names)).to(device)

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "body_fat_percentage_model"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME
IMAGE_PATH = Path("image")
IMAGE_PATH.mkdir(parents=True, exist_ok=True)

model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
model_0.eval()

image_name=input()

custom_image_path = IMAGE_PATH / image_name
custom_image = torchvision.io.read_image(str(custom_image_path)).type(torch.float32)
custom_image=custom_image/255
custom_image_transform = transforms.Compose([
    transforms.Resize((size*4, size*4)),
    transforms.Grayscale()
])

# Transform target image
custom_image_transformed = custom_image_transform(custom_image)

custom_image_transformed = custom_image_transformed.unsqueeze(dim=0)

with torch.inference_mode():
    custom_image_pred = model_0(custom_image_transformed.to(device))

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
custom_image_pred_class = class_names[custom_image_pred_label.cpu()]
print(custom_image_pred_class)
