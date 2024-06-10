# Preprocessing
##
from params import *
from depth_planes.utils import *
from depth_planes.logic.registry import *
from depth_planes.logic.preprocessor import *
from depth_planes.logic.model import *
## Fast
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
## Other
import uuid

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
cache_folder_preprocessed = os.path.join(LOCAL_DATA_PATH, "cache", "preprocessed")
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
app.state.model = load_model()
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

    #print(image_cache_path,image_cache_size,image_cache_extension)

    if not image_cache_extension is in ['jpg','jpeg','png']:
        raise ValueError("Please send an image.")

    if not os.path.exists(image_cache_path):
        return {"error": "Image not found on the server"}

    model = app.state.model
    assert model is not None

    X_processed_path = preprocess_one_image(image_cache_path, cache_folder, 'cache')
    X_processed = get_npy(X_processed_path)[0]
    y_pred = model.predict(X_processed)

    path_pred = save_image(y_pred, cache_folder_preprocessed, filename)

    #return image_cache_path

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(
        url=path_pred, # Url of the depth map
        data=y_pred # The array of the depth map
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

    image_cache_path, image_cache_extension = download_image(url, cache_folder, filename)

    if not image_cache_path.is_file():
        return {"error": "Image not found on the server"}
    return FileResponse(image_path)

    model = app.state.model
    assert model is not None

    X_processed_path = preprocess_one_image(image_cache_path,cache_folder, 'cache')
    X_processed = get_npy(X_processed_path)[0]
    y_pred = model.predict(X_processed)

    path_pred = save_image(y_pred, cache_folder_preprocessed, filename)

    # ⚠️ fastapi only accepts simple Python data types as a return value
    # among them dict, list, str, int, float, bool
    # in order to be able to convert the api response to JSON
    return dict(
        url=list(path_pred), # List of the url of the depth map
        data=y_pred # The array of the depth map
        )
    # $CHA_END

# http://127.0.0.1:8000/convert?url=https://placehold.co/600x400
@app.get("/convert")
async def convert():
    pass

@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END


# $ uvicorn depth_planes.api.fast:app --reload
