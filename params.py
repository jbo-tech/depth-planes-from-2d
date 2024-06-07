import os

##################  VARIABLES  ##################

################## Configuration ##################
IMAGE_ENV=os.environ.get("IMAGE_ENV")
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
CHECKPOINT_PATH=os.path.join(ROOT_DIRECTORY, "saved_files")

##################  CLOUD STORAGE  ##############
GCP_PROJECT=os.environ.get("GCP_PROJECT")
GCP_REGION=os.environ.get("GCP_REGION")
BUCKET_NAME=os.environ.get("BUCKET_NAME")

##################  COMPUTE ENGINE  #############
MODEL_TARGET=os.environ.get("MODEL_TARGET")

print(ROOT_DIRECTORY)
