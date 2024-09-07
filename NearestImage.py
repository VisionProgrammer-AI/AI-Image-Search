import torchvision.models as models
import torch.nn as nn
import torch
from torchvision import transforms
import os
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


resnet50 = models.resnet50(pretrained=True)
resnet50 = nn.Sequential(*list(resnet50.children())[:-1])
resnet50.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

test_image = 'test.jpg'
test_image = Image.open(test_image)
test_image = transform(test_image).unsqueeze(0).to(device)

with torch.no_grad():
    test_features = resnet50(test_image)
    test_features = test_features.view(test_features.size(0),-1).cpu().numpy()

dataset_features = np.load('features.npy')
image_path = np.load('image_path.npy')

distances = euclidean_distances(test_features, dataset_features)

nearest_image_index = np.argmin(distances)
nearest_image_path = image_path[nearest_image_index]

print(f'The nearest image from dataset to test image is: {nearest_image_path}')