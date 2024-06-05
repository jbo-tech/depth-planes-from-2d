import os

##################  VARIABLES  ##################

# DATASET
IMAGE_ENV=os.environ.get("IMAGE_ENV")
DATASET_URBAN=os.environ.get("DATASET_URBAN")
DATASET_MAKE3D=os.environ.get("DATASET_MAKE3")
IMAGE_SHAPE=os.environ.get("IMAGE_SHAPE")

# MODEL
MODEL_TARGET = os.environ.get("MODEL_TARGET")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'),
                                    ".lewagon", "project",
                                    "training_outputs")


##################  CLOUD STORAGE  ##############
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME=os.environ.get("BUCKET_NAME")


##################  COMPUTE ENGINE  #############
