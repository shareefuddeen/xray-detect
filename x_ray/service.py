import torch
import torchxrayvision as xrv
from PIL import Image
import numpy as np

# Set device to CPU
device = torch.device("cpu")

# Load model (DenseNet121, small input 224x224)
model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
model.eval()


def predict_xray(image_file):
    # Open image and convert to grayscale
    img = Image.open(image_file).convert("L")

    # Resize to 224x224 to reduce memory usage
    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)

    # Normalize using torchxrayvision utility
    img = xrv.datasets.normalize(img, 255)

    # Convert to tensor and add batch + channel dimensions
    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    # Run inference
    with torch.no_grad():
        output = model(img_tensor)[0]

    # Convert output to dict with disease labels
    results = {disease: float(prob) for disease, prob in zip(model.pathologies, output)}

    return results
