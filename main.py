import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from depth_planes.logic.model import *
from depth_planes.logic.registry import *
from depth_planes.logic.preprocessor import *
from depth_planes.logic.data import *
from params import *

import logging

logging.basicConfig(filename="depth_planes.log",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

### Check if the save folders exist
if not os.path.exists(LOCAL_REGISTRY_PATH):
    os.makedirs(LOCAL_REGISTRY_PATH)
if not os.path.exists(LOCAL_REGISTRY_MODEL_PATH):
    os.makedirs(LOCAL_REGISTRY_MODEL_PATH)
if not os.path.exists(LOCAL_REGISTRY_METRICS_PATH):
    os.makedirs(LOCAL_REGISTRY_METRICS_PATH)
if not os.path.exists(LOCAL_REGISTRY_PARAMS_PATH):
    os.makedirs(LOCAL_REGISTRY_PARAMS_PATH)
if not os.path.exists(LOCAL_REGISTRY_IMG_PATH):
    os.makedirs(LOCAL_REGISTRY_IMG_PATH)
if not os.path.exists(LOCAL_REGISTRY_CHECKPOINT_PATH):
    os.makedirs(LOCAL_REGISTRY_CHECKPOINT_PATH)

### Load data and preprocess, save preprocessed data on bucket

def preprocess():
    """
    - Query the raw dataset from bucket
    - Process query data
    - Store processed data on new bucket
    """
    preprocess_dataset()

    print("✅ preprocess() done \n")

### Load preprocessed data, train-test-split
def load_processed_data(split_ratio: float = 0.2):

    # Load processed data using data.py

    path_X = f'{LOCAL_DATA_PATH}/ok/_preprocessed/X'
    path_y = f'{LOCAL_DATA_PATH}/ok/_preprocessed/y'

    data_processed_X = get_npy(path_X) #array -> shape (nb, 128,256,3)
    data_processed_y = get_npy(path_y) #array -> shape (nb, 128, 256, 1)

    # Create (X_train, y_train, X_test, y_test)

    X_train, X_test, y_train, y_test = train_test_split(data_processed_X, data_processed_y, test_size=split_ratio)

    print(X_train.shape, y_train.shape)
    print("✅ load_processed_data() done \n")

    return X_train, X_test, y_train, y_test

### Train model
def train(X_train,
          y_train,
          learning_rate=LEARNING_RATE,
          batch_size=BATCH_SIZE,
          patience=PATIENCE,
          validation_split=VALIDATION_SPLIT,
          latent_dimension=LATENT_DIMENSION,
          epochs=EPOCHS):
    """
    - Download processed data from buckets
    - Create train and test splits
    - Train on the preprocessed train dataset
    - Store training results and model weights

    Return val_mae as a float
    """
    # Train model using `model.py`
    model = load_model()

    if model is None:
        encoder = build_encoder(latent_dimension=latent_dimension)
        decoder = build_decoder(latent_dimension=latent_dimension)
        model = build_autoencoder(encoder, decoder)

        model = compile_autoencoder(autoencoder=model, learning_rate=learning_rate)

        model, history = train_model(model=model,
                                    X=X_train,
                                    y=y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    patience=patience,
                                    validation_split=validation_split)

        val_mae = np.min(history.history['val_mae'])

        params = dict(
            context="train", # Package behavior
            training_set_size=len(X_train)
            )

        metrics = dict(context="train",
                    mae=val_mae)

        # Save results on the hard drive using registry.py
        save_results(params=params, metrics=metrics)

        # Save model weight on the hard drive using registry.py
        save_model(model=model)

        print(f"✅ train() done with val_mae : {val_mae}")


### Evaluate model
def evaluate(X_test, y_test, batch_size=64):
    """
    Evaluate the performance of the model on processed test data
    Return MAE as a float
    """
    model = load_model()
    assert model is not None

    metrics_dict = evaluate_model(model=model,
                                  X=X_test,
                                  y=y_test,
                                  batch_size=batch_size)
    mae = metrics_dict["mae"]

    metrics = dict(context="train",
                   mae=mae)

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=len(X_test)
        )

    save_results(params=params, metrics=metrics)

    print("✅ evaluate() done \n")

    return mae

### Make a prediction
def predict(X_pred: np.ndarray = None ) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    model = load_model()
    assert model is not None

    y_pred = model.predict(X_pred)

    print("\n✅ Prediction done: ", y_pred.shape, "\n")

    # Save y_pred locally as an .png image
    save_predicted_image(y_pred=y_pred)

    print("✅ Predicted images saved locally \n")

    return y_pred

if __name__ == '__main__':
    #preprocess()
    X_train, X_test, y_train, y_test = load_processed_data()
    train(X_train=X_train, y_train=y_train)
    evaluate(X_test=X_test, y_test=y_test)
    predict(X_pred=X_test)
