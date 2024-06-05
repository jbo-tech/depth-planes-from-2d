import numpy as np
import pandas as pd

from logic.model import build_autoencoder, build_decoder, build_encoder, compile_autoencoder, train_model, evaluate_model
from logic.registry import save_model, save_results, load_model

### Load data and preprocess, save preprocessed data on bucket

def preprocess():
    """
    - Query the raw dataset from bucket
    - Process query data
    - Store processed data on new bucket
    """

    print("✅ preprocess() done \n")


### Train model
def train(split_ratio: float = 0.2,
        learning_rate=0.001,
        batch_size = 256,
        patience = 2,
        validation_split = 0.2,
        latent_dimension = 32,
        epochs = 500):
    """
    - Download processed data from buckets
    - Create train and test splits
    - Train on the preprocessed train dataset
    - Store training results and model weights

    Return val_mae as a float
    """

    # Load processed data using data.py

    data_processed_X = get_data_from() #array
    data_processed_y = get_data_from() #array

    # Create (X_train, y_train, X_test, y_test)

    train_length = int(len(data_processed_X)*(1-split_ratio))

    X_train = data_processed_X[]
    X_test =

    y_train =
    y_test =


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
        context="evaluate", # Package behavior
        training_set_size=len(X_test)
        )

    # Save results on the hard drive using registry.py
    save_results(params=params, metrics=dict(mae=val_mae))

    # Save model weight on the hard drive using registry.py
    save_model(model=model)

    print("✅ train() done \n")

    return val_mae

### Evaluate model
def evaluate(batch_size=64):
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

    params = dict(
        context="evaluate", # Package behavior
        training_set_size=len(X_test)
        )

    save_results(params=params, metrics=metrics_dict)

    print("✅ evaluate() done \n")

    return mae

### Make a prediction
def predict(X_pred: np.ndarray = None ) -> np.ndarray:
    """
    Make a prediction using the latest trained model
    """
    model = load_model()
    assert model is not None

    #X_processed =

    y_pred = model.predict(X_processed)

    print("\n✅ prediction done: ", y_pred.shape, "\n")

    return y_pred

if __name__ == '__main__':
    preprocess()
    train()
    evaluate()
    predict()
