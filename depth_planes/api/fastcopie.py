# Preprocessing
##
from params import *
from main import predict_and_save_DPTForDepthEstimation
from depth_planes.utils import *
from depth_planes.logic.registry import *
from depth_planes.logic.preprocessor import *
from depth_planes.logic.model import *
## Fast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
## Other
import uuid
import json
from fastapi import FastAPI, UploadFile, File, HTTPException
import cv2


app = FastAPI()

### Check if the necessary folders exist
cache_folder = os.path.join(LOCAL_DATA_PATH, "cache")
cache_folder_preprocessed = os.path.join(LOCAL_DATA_PATH, "cache", "_preprocessed")
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)
if not os.path.exists(cache_folder_preprocessed):
    os.makedirs(cache_folder_preprocessed)

@app.get("/depth")
async def depth(
        img: UploadFile=File(...)
    ):
    model = app.state.model
    contents = await img.read()
    nparr = np.fromstring(contents, np.uint8)
    cv2_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    X_processed_path = preprocess_one_image(cv2_img, cache_folder, 'cache')
    # used for the futur model
    X_processed = np.expand_dims(get_npy_direct(X_processed_path),axis=0)
    #prev with the transformer
    y_pred, pred_img_path = predict_and_save_DPTForDepthEstimation(cv2_img, path=cache_folder_preprocessed)
    return {
        "pred":y_pred.tolist() # The array of the depth map
    }


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Big up to the team")
    # $CHA_END


