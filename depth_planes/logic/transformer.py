import torch 
from PIL import Image
from transformers import DPTImageProcessor, DPTForDepthEstimation
import time
import os 
import numpy as np
from params import *

### Make a prediction with Hugging Face model
def predict_and_save_DPTForDepthEstimation(image_path, path=LOCAL_REGISTRY_IMG_PATH): # --> returns pred_img_path

    #image_url = "/Users/leslierolland/code/soapoperator/depth-planes-from-2d/raw_data/photo_test/urbansyn_rgb_rgb_0034.png"
    image = Image.open(image_path)

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

    # Save y_pred locally as an .png image
    name = time.strftime("%Y%m%d-%H%M%S")
    pred_img_path=os.path.join(path, f"{name}.png")
    depth.save(pred_img_path, format='PNG')

    print("\n✅ Prediction done: ", formatted.shape, "\n")

    return formatted, pred_img_path