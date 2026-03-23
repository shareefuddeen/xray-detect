import torch

torch.set_num_threads(1)
import torchxrayvision as xrv
from PIL import Image
import numpy as np

device = torch.device("cpu")

model = xrv.models.DenseNet(weights="densenet121-res224-all").to(device)
model.eval()


def predict_xray(image_file):
    img = Image.open(image_file).convert("L")

    img = img.resize((224, 224))
    img = np.array(img, dtype=np.float32)

    img = xrv.datasets.normalize(img, 255)

    img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)[0]

    results = {disease: float(prob) for disease, prob in zip(model.pathologies, output)}

    return results
