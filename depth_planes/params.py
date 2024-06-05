import os
# Configuration
IMAGE_ENV=os.environ.get("IMAGE_ENV")
DATA_URBAN=os.environ.get("DATA_URBAN")
DATA_MAKE3D=os.environ.get("DATA_MAKE3")
DATA_DIODE=os.environ.get("DATA_DIODE")
DATA_MEGADEPTH=os.environ.get("DATA_MEGADEPTH")
DATA_DIMLRGBD=os.environ.get("DATA_DIMLRGBD")
DATA_NYUDEPTHV2=os.environ.get("DATA_NYUDEPTHV2")
IMAGE_SHAPE=os.environ.get("IMAGE_SHAPE")

##################  VARIABLES  ##################
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")

# Cloud Storage
BUCKET_NAME=os.environ.get("BUCKET_NAME")

# Compute Engine

# Local
LOCAL_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "raw_data")
