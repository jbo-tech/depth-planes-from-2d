import numpy as np
from transformers import DPTImageProcessor, DPTForDepthEstimation
import torch
from PIL import Image

from depth_planes.logic.model import *
from depth_planes.logic.registry import *
from depth_planes.logic.preprocessor import *
from depth_planes.logic.data import *
from params import *


def predict_DPTForDepthEstimation(image): # --> returns pred_img

    # image = np.fromstring(image,np.uint8)
    # image = cv2.imdecode(image,cv2.IMREAD_COLOR)

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    # prepare image for the model
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)
    depth.save("here.png")
    print("\n✅ Prediction done: ", formatted.shape, "\n")
