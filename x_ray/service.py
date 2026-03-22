import torch
import torchxrayvision as xrv
from PIL import Image
import numpy as np

# Load ResNet18 once globally
model = xrv.models.ResNet(weights="resnet18-res224-all")
model.eval()  # set model to evaluation mode


def predict_xray(image_file):
    """
    Predict X-ray diseases from uploaded image.
    Returns a list of predictions.
    """
    # Open and resize image
    img = Image.open(image_file).convert("L")
    img = img.resize((224, 224))  # ResNet18 expects 224x224

    # Convert to numpy and normalize
    img = np.array(img, dtype=np.float32)
    img = xrv.datasets.normalize(img, 255)
    img = np.expand_dims(img, 0)  # add batch dimension
    img = np.expand_dims(img, 0)  # add channel dimension (1,1,224,224)

    # Predict
    with torch.no_grad():
        tensor = torch.tensor(img, dtype=torch.float32)
        output = model(tensor)

    return output[0].tolist()
