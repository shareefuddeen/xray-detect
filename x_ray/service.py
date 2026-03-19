# detector/services.py

import torch
import torchxrayvision as xrv
import numpy as np
from PIL import Image
import torch.nn.functional as F

torch.backends.mkldnn.enabled = False

model = None  # global


def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        model.eval()
    return model


def predict_xray(image_file):
    img = Image.open(image_file).convert("L")
    img = np.array(img)
    img = xrv.datasets.normalize(img, 255)

    model = get_model()

    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()
    img = F.interpolate(img, size=(224, 224), mode="bilinear", align_corners=False)

    with torch.no_grad():
        output = model(img)[0]

    results = list(zip(xrv.datasets.default_pathologies, output))
    results = sorted(results, key=lambda x: float(x[1]), reverse=True)

    tb_score = 0
    cardio_score = 0

    for p, s in results:
        s = float(s)
        if p in ["Infiltration", "Consolidation"]:
            tb_score = max(tb_score, s)
        elif p == "Cardiomegaly":
            cardio_score = max(cardio_score, s)

    filtered = []
    if tb_score > 0.6:
        filtered.append({"disease": "Tuberculosis", "confidence": tb_score})
    elif cardio_score > 0.5:
        filtered.append({"disease": "Cardiomegaly", "confidence": cardio_score})

    return filtered
