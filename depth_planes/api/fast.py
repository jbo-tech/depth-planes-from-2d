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

# http://127.0.0.1:8000/depth?url=https://placehold.co/600x400
@app.get("/depth")
async def depth(
        url: str,  # https://placehold.co/600x400
    ):
    """
    Return a depth map of an image.
    Assumes `url` is provided.
    """
    # $CHA_BEGIN

    filename = str(uuid.uuid4())

    image_cache_path, image_cache_extension = download_image(url, cache_folder, filename)
    image_cache_size = get_image_size(image_cache_path)

    print(image_cache_path,image_cache_size,image_cache_extension)

    if image_cache_extension[1:] not in ['jpg','jpeg','png']:
        raise ValueError("Please send an image.")

    if not os.path.exists(image_cache_path):
        return {"error": "Image not found on the server"}

    if API_MODEL == 'local':

        model = app.state.model
        assert model is not None

        X_processed_path = preprocess_one_image(image_cache_path, cache_folder, 'cache')
        X_processed = np.expand_dims(get_npy_direct(X_processed_path),axis=0)
        #print(X_processed.shape)
        y_pred = model.predict(X_processed)
        #print(y_pred)

        all_path_pred = save_image(y_pred, cache_folder_preprocessed, filename)
        pred_img_path = [x for x in all_path_pred if x.startswith(filename)][0]

    elif API_MODEL == 'hf':

        y_pred, pred_img_path = predict_and_save_DPTForDepthEstimation(image_cache_path, path=cache_folder_preprocessed)

        #print(type(y_pred),y_pred.shape)
        #print(pred_img_path)

    bb = upload_one_file(pred_img_path,filename,pred_img_path.split('.')[-1],'_cache')
    #print(bb)

    url = get_blob_url(bb)
    #print(url)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(
        url=url, # Url of the depth map
        data=json.dumps(y_pred.tolist()) # The array of the depth map
        )

    # $CHA_END

# http://127.0.0.1:8000/slice?url=https://placehold.co/600x400&nb_planes=10
@app.get("/slices")
async def slices(
        url: str,  # https://placehold.co/600x400get image size  file path
        nb_planes: int | None = 5 # 1 or None
    ):
    """
    Return all the planes of an image.
    Assumes `url` is provided.
    """
    # $CHA_BEGIN

    filename = str(uuid.uuid4())

    image_cache_path, image_cache_extension = download_image(url, cache_folder + '_' + filename, filename)
    image_cache_size = get_image_size(image_cache_path)

    print(image_cache_path,image_cache_size,image_cache_extension)

    if image_cache_extension[1:] not in ['jpg','jpeg','png']:
        raise ValueError("Please send an image.")

    if not os.path.exists(image_cache_path):
        return {"error": "Image not found on the server"}

    if API_MODEL == 'local':

        model = app.state.model
        assert model is not None

        X_processed_path = preprocess_one_image(image_cache_path, cache_folder, 'cache')
        X_processed = np.expand_dims(get_npy_direct(X_processed_path),axis=0)
        #print(X_processed.shape)
        y_pred = model.predict(X_processed)
        #print(y_pred)

        all_path_pred = save_image(y_pred, cache_folder_preprocessed, filename)
        pred_img_path = [x for x in all_path_pred if x.startswith(filename)][0]

    elif API_MODEL == 'hf':

        y_pred, pred_img_path = predict_and_save_DPTForDepthEstimation(image_cache_path, path=cache_folder_preprocessed)

        #print(type(y_pred),y_pred.shape)
        #print(pred_img_path)

    bb = upload_one_file(pred_img_path,filename,pred_img_path.split('.')[-1],'_cache')
    #print(bb)

    url = get_blob_url(bb)
    #print(url)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(
        url=url, # Url of the depth map
        data=json.dumps(y_pred.tolist()) # The array of the depth map
        )

# http://127.0.0.1:8000/convert?url=https://placehold.co/600x400
@app.get("/convert")
async def convert(
        url: str,  # https://placehold.co/600x400
    ):
    """
    Return a depth map as an array.
    Assumes `url` is provided.
    """
    # $CHA_BEGIN

    filename = str(uuid.uuid4())

    image_cache_path, image_cache_extension = download_image(url, cache_folder, filename)
    image_cache_size = get_image_size(image_cache_path)

    #print(image_cache_path,image_cache_size,image_cache_extension)

    if image_cache_extension[1:] not in ['jpg','jpeg','png']:
        raise ValueError("Please send an image.")

    if not os.path.exists(image_cache_path):
        return {"error": "Image not found on the server"}

    if API_MODEL == 'local':

        model = app.state.model
        assert model is not None

        X_processed_path = preprocess_one_image(image_cache_path, cache_folder, 'cache')
        X_processed = np.expand_dims(get_npy_direct(X_processed_path),axis=0)
        #print(X_processed.shape)
        y_pred = model.predict(X_processed)
        #print(y_pred)

        all_path_pred = save_image(y_pred, cache_folder_preprocessed, filename)
        pred_img_path = [x for x in all_path_pred if x.startswith(filename)][0]

    elif API_MODEL == 'hf':

        y_pred, pred_img_path = predict_and_save_DPTForDepthEstimation(image_cache_path, path=cache_folder_preprocessed)

        #print(type(y_pred),y_pred.shape)
        #print(pred_img_path)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(
        data=json.dumps(y_pred.tolist()) # The array of the depth map
        )

    # $CHA_END


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Big up to the team")
    # $CHA_END


# $ uvicorn depth_planes.api.fast:app --reload
