import torchvision.models as models
import torch.nn as nn
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np

resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
resnet50.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

data_directory = 'Dataset'

all_features = []
all_image_path = []

for image_name in os.listdir(data_directory):
    image_path = os.path.join(data_directory, image_name)

    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = resnet50(image)
        features = features.view(features.size(0),-1).cpu().numpy()

        all_features.append(features)
        all_image_path.append(image_path)

all_features = np.vstack(all_features)
all_image_path = np.array(all_image_path)

np.save('features.npy', all_features)
np.save('image_path.npy', all_image_path)
