# data_processing/feature_extraction.py

import torch
from torchvision import models, transforms

def extract_features(images):
    # Use a pre-trained model
    model = models.resnet50(pretrained=True)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove last layer
    model.eval()

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])

    features = []
    with torch.no_grad():
        for image in images:
            input_tensor = preprocess(image).unsqueeze(0)
            output = model(input_tensor)
            features.append(output.squeeze().numpy())
    return features
