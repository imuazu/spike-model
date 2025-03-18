import pandas as pd
import numpy as np
import argparse
from keras.models import Model
from keras.layers import Input, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Dense, Add, Activation
from sklearn.model_selection import train_test_split
from src.utils.preprocess import preprocess_data

def identity_block(x, filters):
    shortcut = x
    x = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv1D(filters, kernel_size=3, padding='same')(x)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def create_resnet_model(input_shape):
    inputs = Input(shape=input_shape)
    x = Conv1D(64, kernel_size=7, padding='same', activation='relu')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    for _ in range(3):
        x = identity_block(x, 64)
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResNet50 Model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset")
    args = parser.parse_args()

    # Load and preprocess data
    data = pd.read_csv(args.data)
    X, y_one_hot = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    # Reshape for ResNet
    X_train_resnet = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_resnet = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Train ResNet
    resnet_model = create_resnet_model((X_train.shape[1], 1))
    resnet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    resnet_model.fit(X_train_resnet, y_train, epochs=10, batch_size=32, verbose=1)
    y_proba = resnet_model.predict(X_test_resnet)

    # Save results
    np.save("results/resnet_proba.npy", y_proba)