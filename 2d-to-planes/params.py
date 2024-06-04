import os


##################  VARIABLES  ##################
GCP_PROJECT = os.environ.get("GCP_PROJECT")
GCP_REGION = os.environ.get("GCP_REGION")
BUCKET_NAME = os.environ.get("BUCKET_NAME")
MODEL_TARGET = os.environ.get("MODEL_TARGET")
LOCAL_REGISTRY_PATH =  os.path.join(os.path.expanduser('~'),
                                    ".lewagon", "project",
                                    "training_outputs")
