import os
import numpy as np
import soundfile as sf
import tensorflow as tf
from tensorflow.keras import layers, Model

DATA_DIR = "data_wavs"
FRAME_SIZE = 2048
HOP = 1024
EPOCHS = 25
BATCH = 16


# ------------------------------------------------------
# Load WAV dataset
# ------------------------------------------------------
def load_dataset(path):
    all_frames = []

    for fname in os.listdir(path):
        if not fname.lower().endswith(".wav"):
            continue

        wav, sr = sf.read(os.path.join(path, fname))
        if sr != 16000:
            raise ValueError(f"{fname} is not 16kHz!")

        # Normalize
        wav = wav / np.max(np.abs(wav))

        # Slice to frames
        for i in range(0, len(wav) - FRAME_SIZE, HOP):
            frame = wav[i:i+FRAME_SIZE]
            all_frames.append(frame)

    all_frames = np.array(all_frames, dtype=np.float32)
    print(f"Loaded {len(all_frames)} frames from {path}")
    return all_frames


# ------------------------------------------------------
# Build Autoencoder (1D CNN)
# ------------------------------------------------------
def build_autoencoder():
    inp = layers.Input(shape=(FRAME_SIZE, 1))

    x = layers.Conv1D(32, 9, activation="relu", padding="same", strides=2)(inp)
    x = layers.Conv1D(64, 9, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv1D(128, 9, activation="relu", padding="same", strides=2)(x)

    # Bottleneck
    x = layers.Conv1D(256, 9, activation="relu", padding="same")(x)

    # Decoder
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(128, 9, activation="relu", padding="same")(x)

    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64, 9, activation="relu", padding="same")(x)

    x = layers.UpSampling1D(2)(x)
    out = layers.Conv1D(1, 9, activation="tanh", padding="same")(x)

    model = Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    model.summary()
    return model


# ------------------------------------------------------
# Save sample reconstruction
# ------------------------------------------------------
def save_reconstruction(model, frame, sr=16000):
    pred = model.predict(frame[np.newaxis, :, np.newaxis])[0, :, 0]
    pred = pred / np.max(np.abs(pred))
    sf.write("sample_reconstruction.wav", pred, sr)
    print("Saved sample_reconstruction.wav")


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------
if __name__ == "__main__":
    print("Loading dataset...")
    frames = load_dataset(DATA_DIR)

    # Prepare for training
    frames = frames[..., np.newaxis]  # add channel
    print("Shape:", frames.shape)

    # Shuffle & split
    total = len(frames)
    val_split = int(0.9 * total)
    train = frames[:val_split]
    val = frames[val_split:]

    print(f"Train: {len(train)}, Val: {len(val)}")

    # Build model
    print("Building autoencoder...")
    ae = build_autoencoder()

    # Train
    print("Training...")
    ae.fit(
        train, train,
        validation_data=(val, val),
        epochs=EPOCHS,
        batch_size=BATCH
    )

    # Save model
    ae.save("trained_autoencoder.h5")
    print("Saved trained_autoencoder.h5")

    # Save recon sample
    save_reconstruction(ae, frames[0, :, 0])

    print("\nTraining complete.")
