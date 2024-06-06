import numpy as np
from keras import models, layers, optimizers, losses, metrics, Sequential, Model, callbacks

def build_encoder(latent_dimension):
    '''
    Returns an encoder model, of output_shape equals to latent_dimension
    '''
    encoder = Sequential([layers.Input((128,256,3)),
                                 layers.Conv2D(8, (2,2), activation='relu'),
                                 layers.MaxPool2D(2),
                                 layers.Conv2D(16, (2,2), activation='relu'),
                                 layers.MaxPool2D(2),
                                 layers.Conv2D(32, (2,2), activation='relu'),
                                 layers.MaxPool2D(2),
                                 layers.Flatten(),
                                 layers.Dense(latent_dimension, activation='tanh')
                                 ])

    print("✅ Encoder built")
    return encoder


def build_decoder(latent_dimension):
    '''
    Returns an decoder model, with outputs images of same shape than the encoder input
    '''
    decoder = Sequential([layers.Dense(64*32*8, activation='tanh', input_shape=(latent_dimension,)),
                                 layers.Reshape((32, 64, 8)),
                                 layers.Conv2DTranspose(8, (2, 2), strides=2, padding='same', activation='relu'),
                                 layers.Conv2DTranspose(1, (2, 2), strides=2, padding='same', activation='relu')
                                ])
    print("✅ Decoder built")
    return decoder


def build_autoencoder(encoder, decoder):
    '''
    Building the autoencoder
    '''
    inp = layers.Input((128,256,3))
    encoded = encoder(inp)
    decoded = decoder(encoded)
    autoencoder = Model(inp, decoded)

    print("✅ Autoencoder built")

    return autoencoder


def compile_autoencoder(autoencoder, learning_rate=0.001):
    """
    Compile the Auto Encoder
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)
    autoencoder.compile(loss='mae',
                  optimizer=optimizer,
                  metrics=['mae'])

    print("✅ Model compiled")

    return autoencoder


def train_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=256,
        patience=10,
        epochs=500,
        validation_data=None, # overrides validation_split
        validation_split=0.3):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=1
    )

    print(f"✅ Model trained on {len(X)} images")

    return model, history


def evaluate_model(
        model: Model,
        X: np.ndarray,
        y: np.ndarray,
        batch_size=64
        ):
    """
    Evaluate trained model performance on the dataset
    """
    metrics = model.evaluate(
        x=X,
        y=y,
        batch_size=batch_size,
        verbose=0,
        callbacks=None,
        return_dict=True
    )

    loss = metrics["loss"]
    mae = metrics["mae"]

    print(f"✅ Model evaluated, MAE: {round(mae, 2)}")

    return metrics
