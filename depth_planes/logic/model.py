import numpy as np
from keras import models, layers, optimizers, losses, metrics, Sequential, Model, callbacks
from params import *

def build_encoder(latent_dimension):
    '''
    Returns an encoder model, of output_shape equals to latent_dimension
    '''
    encoder = Sequential([layers.Input((128,256,3)),
                                 layers.Conv2D(32, (2,2), activation='relu'),
                                 layers.MaxPool2D(2),
                                 layers.Conv2D(16, (2,2), activation='relu'),
                                 layers.MaxPool2D(2),
                                 layers.Conv2D(8, (2,2), activation='relu'),
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

def build_model(input_shape=(128, 256, 3)):
    inputs = layers.Input(shape=input_shape)

    ### Encoder
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)

    ### Decoder
    u6 = layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model


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
        batch_size=1,
        patience=50,
        epochs=3000,
        validation_data=None, # overrides validation_split
        validation_split=0.2):
    """
    Fit the model and return a tuple (fitted_model, history)
    """
    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    checkpoint_filepath=os.path.join(LOCAL_REGISTRY_CHECKPOINT_PATH, 'checkpoints', 'checkpoint.model.keras')

    cp = callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        save_best_only=True,
        save_freq='epoch'
    )

    history = model.fit(
        X,
        y,
        validation_data=validation_data,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es, cp],
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
