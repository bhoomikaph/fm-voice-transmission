"""
fm_voice_simulation_with_nn.py
Final stable FM simulation + neural autoencoder
With pre-emphasis, deemphasis, lowpass stabilization, and improved AE training.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import librosa
from scipy.signal import hilbert, butter, filtfilt
from numpy.random import normal
from tensorflow.keras import layers, models, callbacks
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# Utility
# =========================================================
def normalize(x):
    maxv = np.max(np.abs(x))
    return x if maxv == 0 else x / maxv

def compute_snr(orig, recon):
    L = min(len(orig), len(recon))
    orig = orig[:L]
    recon = recon[:L]
    sig = np.mean(orig**2)
    noise = np.mean((orig - recon)**2) + 1e-12
    return 10 * np.log10(sig / noise)

def lowpass(sig, cutoff, fs, order=6):
    b, a = butter(order, cutoff/(0.5*fs), btype='low')
    return filtfilt(b, a, sig)

def preemphasis(sig, p=0.97):
    """Push high freq before FM → improves clarity."""
    return np.append(sig[0], sig[1:] - p * sig[:-1])

def deemphasis(sig, fs, tau=75e-6):
    """Reverse pre-emphasis after FM demod."""
    alpha = fs * tau
    out = np.zeros_like(sig)
    for i in range(1, len(sig)):
        out[i] = out[i-1] + (sig[i] - out[i-1]) / (1 + alpha)
    return out

# =========================================================
# Load audio
# =========================================================
infile = "input_voice.wav"
if not os.path.exists(infile):
    raise FileNotFoundError("Place 'input_voice.wav' in this folder.")

y, sr = librosa.load(infile, sr=16000, mono=True)
y = y.astype(np.float32)
y = normalize(y)

print(f"Loaded 'input_voice.wav' — {sr} Hz, duration: {len(y)/sr:.2f} s")

# bandlimit input
y = lowpass(y, 4000, sr)
y = normalize(y)

# apply pre-emphasis
y_pre = preemphasis(y)
y_pre = normalize(y_pre)

# =========================================================
# FM FUNCTIONS
# =========================================================
def fm_modulate(message, fs, fc, kf):
    int_msg = np.cumsum(message) / fs
    t = np.arange(len(message)) / fs
    phase = 2*np.pi*(fc*t + kf*int_msg)
    return np.cos(phase)

def awgn(signal, snr_db):
    sp = np.mean(signal**2)
    npow = sp / (10**(snr_db/10))
    noise = np.sqrt(npow) * normal(0, 1, len(signal))
    return signal + noise

def fm_demodulate(received, fs, fc, kf):
    analytic = hilbert(received)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) * fs/(2*np.pi)

    demod = (inst_freq - fc) / kf
    demod = np.concatenate([demod, [demod[-1]]])
    demod = lowpass(demod, 3800, fs)
    demod = deemphasis(demod, fs)
    return normalize(demod)

# =========================================================
# FM PARAMS
# =========================================================
fc = 1800
kf = 120
SNR_dB = 45      # cleaner FM channel for neural path

# =========================================================
# ANALOG FM CHAIN
# =========================================================
print("\nRunning ANALOG FM chain...")

fm_tx = fm_modulate(y_pre, sr, fc, kf)
noisy_fm = awgn(fm_tx, SNR_dB)
recovered_analog = fm_demodulate(noisy_fm, sr, fc, kf)
recovered_analog = normalize(recovered_analog)

sf.write("recovered_analog.wav", recovered_analog, sr)
print("Saved recovered_analog.wav")

# =========================================================
# Autoencoder FRAME SETUP
# =========================================================
frame_len = 1024
hop = 512

def frame_signal(sig, frame_len, hop):
    return librosa.util.frame(sig, frame_length=frame_len, hop_length=hop).T

frames = frame_signal(y_pre, frame_len, hop).astype(np.float32)
frames = frames[..., np.newaxis]
frames /= (np.max(np.abs(frames)) + 1e-9)

n = len(frames)
train = frames[:int(0.9*n)]
val   = frames[int(0.9*n):]

print(f"Frames: {n}, train: {len(train)}, val: {len(val)}")

# =========================================================
# AUTOENCODER
# =========================================================
def build_autoencoder():
    inp = layers.Input(shape=(frame_len,1))

    # Encoder
    x = layers.Conv1D(32, 9, strides=2, padding='same', activation='relu')(inp)
    x = layers.Conv1D(64, 7, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(128,5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv1D(256,5, padding='same', activation='relu')(x)

    # Decoder
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(128,5,padding='same',activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(64,7,padding='same',activation='relu')(x)
    x = layers.UpSampling1D(2)(x)
    x = layers.Conv1D(32,9,padding='same',activation='relu')(x)

    out = layers.Conv1D(1,9,padding='same',activation='linear')(x)

    return models.Model(inp, out)

autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()

# Early stopping
es = callbacks.EarlyStopping(patience=10, restore_best_weights=True)

print("\nTraining neural codec (autoencoder)...")
autoencoder.fit(train, train,
                epochs=60,
                batch_size=32,
                validation_data=(val, val),
                callbacks=[es],
                verbose=2)

# =========================================================
# Reconstruct audio
# =========================================================
reconstructed = autoencoder.predict(frames, batch_size=32)
reconstructed = reconstructed.squeeze(-1)

def overlap_add(frames, hop):
    L = frames.shape[1]
    out_len = (len(frames)-1)*hop + L
    out = np.zeros(out_len)
    win = np.hanning(L)
    wsum = np.zeros(out_len)

    for i,f in enumerate(frames):
        s = i*hop
        out[s:s+L] += f * win
        wsum[s:s+L] += win**2

    return out / (wsum + 1e-12)

recon_audio = overlap_add(reconstructed, hop)
recon_audio = lowpass(recon_audio, 3800, sr)   # IMPORTANT
recon_audio = normalize(recon_audio)

sf.write("neural_reconstructed.wav", recon_audio, sr)
print("Saved neural_reconstructed.wav")

# =========================================================
# NEURAL FM CHAIN
# =========================================================
print("\nRunning Neural Codec + FM chain...")

fm_tx_nn = fm_modulate(recon_audio, sr, fc, kf)
noisy_nn = awgn(fm_tx_nn, SNR_dB)
recovered_nn = fm_demodulate(noisy_nn, sr, fc, kf)
recovered_nn = normalize(recovered_nn)

sf.write("recovered_neural.wav", recovered_nn, sr)
print("Saved recovered_neural.wav")

# =========================================================
# Evaluation
# =========================================================
snr_analog = compute_snr(y, recovered_analog)
snr_neural = compute_snr(y, recovered_nn)

print(f"\nAnalog FM recovered SNR: {snr_analog:.2f} dB")
print(f"Neural Codec + FM recovered SNR: {snr_neural:.2f} dB")

# =========================================================
# Plots
# =========================================================
plt.figure(figsize=(12,8))
plt.subplot(3,1,1); plt.title("Original"); plt.plot(y)
plt.subplot(3,1,2); plt.title("Analog Recovered"); plt.plot(recovered_analog)
plt.subplot(3,1,3); plt.title("Neural Recovered"); plt.plot(recovered_nn)
plt.tight_layout()
plt.savefig("waveforms_comparison.png")
print("Saved waveforms_comparison.png")

print("\nDone! Generated files:")
print(" - recovered_analog.wav")
print(" - neural_reconstructed.wav")
print(" - recovered_neural.wav")
print(" - waveforms_comparison.png")
# =========================================================
# Save SNR comparison to file
# =========================================================
with open("snr_report.txt", "w") as f:
    f.write("FM Voice Transmission SNR Comparison\n")
    f.write("-----------------------------------\n")
    f.write(f"Original → Analog FM SNR      : {snr_analog:.2f} dB\n")
    f.write(f"Original → Neural Codec + FM  : {snr_neural:.2f} dB\n")

print("\nSaved snr_report.txt")
