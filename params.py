import os

##################  VARIABLES  ##################

################## Configuration ##################
IMAGE_ENV=os.environ.get("IMAGE_ENV")
SAVE_GCS=os.environ.get("SAVE_GCS")
DATA_URBANSYN=os.environ.get("DATA_URBANSYN")
DATA_MAKE3D=os.environ.get("DATA_MAKE3D")
DATA_DIODE=os.environ.get("DATA_DIODE")
DATA_MEGADEPTH=os.environ.get("DATA_MEGADEPTH")
DATA_DIMLRGBD=os.environ.get("DATA_DIMLRGBD")
DATA_NYUDEPTHV2=os.environ.get("DATA_NYUDEPTHV2")
IMAGE_SHAPE=os.environ.get("IMAGE_SHAPE")

################## Local ##################
ROOT_DIRECTORY=os.path.dirname(os.path.realpath(__file__))
LOCAL_DATA_PATH=os.path.join(ROOT_DIRECTORY, "raw_data")
LOCAL_REGISTRY_PATH=os.path.join(ROOT_DIRECTORY, "saved_files")
LOCAL_REGISTRY_MODEL_PATH=os.path.join(LOCAL_REGISTRY_PATH, 'models')
LOCAL_REGISTRY_METRICS_PATH=os.path.join(LOCAL_REGISTRY_PATH, 'metrics')
LOCAL_REGISTRY_PARAMS_PATH=os.path.join(LOCAL_REGISTRY_PATH, 'params')
LOCAL_REGISTRY_IMG_PATH=os.path.join(LOCAL_REGISTRY_PATH, 'predicted_img')
LOCAL_REGISTRY_CHECKPOINT_PATH=os.path.join(LOCAL_REGISTRY_PATH, 'checkpoints')

##################  CLOUD STORAGE  ##############
GCP_PROJECT_OLD=os.environ.get("GCP_PROJECT_OLD")
GCP_REGION_OLD=os.environ.get("GCP_REGION_OLD")
BUCKET_NAME_OLD=os.environ.get("BUCKET_NAME_OLD")

GCP_PROJECT=os.environ.get("GCP_PROJECT")
GCP_REGION=os.environ.get("GCP_REGION")
BUCKET_NAME=os.environ.get("BUCKET_NAME")

##################  COMPUTE ENGINE  #############
MODEL_TARGET=os.environ.get("MODEL_TARGET")

##################  MODEL  #############
LEARNING_RATE=float(os.environ.get("LEARNING_RATE"))
BATCH_SIZE=int(os.environ.get("BATCH_SIZE"))
PATIENCE=int(os.environ.get("PATIENCE"))
VALIDATION_SPLIT=float(os.environ.get("VALIDATION_SPLIT"))
LATENT_DIMENSION=int(os.environ.get("LATENT_DIMENSION"))
EPOCHS=int(os.environ.get("EPOCHS"))

##################  API  #############
API_MODEL=os.environ.get("API_MODEL")
