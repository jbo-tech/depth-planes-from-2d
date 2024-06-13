# Preprocessing
##
from params import *
from main import predict_and_save_DPTForDepthEstimation
from depth_planes.utils import *
from depth_planes.logic.registry import *
from depth_planes.logic.preprocessor import *
from depth_planes.logic.model import *
from depth_planes.logic.predict import *
from depth_planes.logic.mask import *
## Fast
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
## Other
import uuid
import json
import cv2
from PIL import Image
import io
from tensorflow.keras.preprocessing.image import img_to_array, load_img


app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

### Check if the necessary folders exist
cache_folder = os.path.join(LOCAL_DATA_PATH, "cache")
cache_folder_preprocessed = os.path.join(LOCAL_DATA_PATH, "cache", "_preprocessed")
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)
if not os.path.exists(cache_folder_preprocessed):
    os.makedirs(cache_folder_preprocessed)

# $WIPE_BEGIN
# 💡 Preload the model to accelerate the predictions
# We want to avoid loading the heavy Deep Learning model from MLflow at each `get("/predict")`
# The trick is to load the model in memory when the Uvicorn server starts
# and then store the model in an `app.state.model` global variable, accessible across all routes!
# This will prove very useful for the Demo Day
# app.state.model = load_model()
# $WIPE_END

@app.post("/depthmap")
async def depthmap(
        file: UploadFile=File(...)  # https://placehold.co/600x400
    ):
    """
    Return a depth map of an image.
    Assumes `url` is provided.
    """
    # $CHA_BEGIN

    filename = str(uuid.uuid4())
    file = await file.read()
    # # file_ = np.fromstring(file_,np.uint8)
    # # file_ = cv2.imdecode(file_,cv2.IMREAD_COLOR)

    image = Image.open(io.BytesIO(file))
    png_format = io.BytesIO()
    image.save(png_format, format='PNG')
    png_format.seek(0)

    predict_DPTForDepthEstimation(image)

    # img_byte_arr = io.BytesIO()
    # depth_pred.save(img_byte_arr, format='PNG')
    # img_byte_arr.seek(0)
    # img_byte_arr = img_byte_arr.getvalue()
    # print(type(img_byte_arr))
    # print(type(img_byte_arr))
    # print(img_byte_arr)
    # ⚠️ fastaimg_byte_arrpi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON

    return FileResponse("here.png")
    # $CHA_END

# http://127.0.0.1:8000/slice?url=https://placehold.co/600x400&nb_planes=10
@app.post("/slices")
async def depthmap(
        file: UploadFile=File(...),
        depth: UploadFile=File(...)
    ):
    """
    Return a depth map of an image.
    Assumes `url` is provided.
    """
    # $CHA_BEGIN

    filename = str(uuid.uuid4())
    file = await file.read()
    depth = await depth.read()

    image = Image.open(io.BytesIO(file))
    png_format = io.BytesIO()
    image.save(png_format, format='PNG')
    png_format.seek(0)

    y_pred = img_to_array(depth)
    print(y_pred)

    mask = create_mask_in_one(y_pred, nb_mask=5)
    #plans_array = create_mask_from_image(x_path :str, y_path: str, y_prec_path)

    return dict(plans='test')
    # $CHA_END

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Big up to the team!")
    # $CHA_END


# $ uvicorn depth_planes.api.fast:app --reload
