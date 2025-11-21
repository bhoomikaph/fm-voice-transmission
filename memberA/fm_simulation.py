# fm_simulation.py
# FM evaluation pipeline (analog vs neural codec) — corrected with upsampling to avoid aliasing
# Place in memberA/ with trained_autoencoder.h5 and data_wavs/
# Run:
#   .\venv\Scripts\activate
#   python fm_simulation.py

import os
import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt
from scipy.signal import hilbert, butter, filtfilt, get_window
from numpy.random import default_rng
import tensorflow as tf

# -----------------------
# Paths & params
# -----------------------
DATA_DIR = "data_wavs"
MODEL_PATH = "trained_autoencoder.h5"
OUT_DIR = "outputs"
SR = 16000                       # audio sample rate used for AE and final files
FS_MOD = 48000                   # internal modulation sample rate (must be > 2*(carrier+deviation))
rng = default_rng()
os.makedirs(OUT_DIR, exist_ok=True)

# SNR levels to evaluate
SNR_LEVELS = [30, 15, 5]

# FM parameters (operate at FS_MOD)
FC = 15000.0    # carrier freq (Hz) — now safe with FS_MOD=48k
KF = 5000.0     # frequency sensitivity (Hz per amplitude-unit)
LOWPASS_CUT_AUDIO = 4000        # lowpass for message before modulation (audio BW)
LOWPASS_CUT_DEMOD = 4000       # demod lowpass cutoff in Hz (audio-band)
FS_MOD_ANTI_ALIAS = 18000      # lowpass cutoff before resampling to FS_MOD (if needed)

# Framing (must match your AE)
FRAME = 2048
HOP = 1024

# -----------------------
# Helpers
# -----------------------
def normalize(x):
    if x is None:
        return x
    x = np.asarray(x, dtype=np.float32)
    m = np.max(np.abs(x))
    if m < 1e-12:
        return x
    return x / (m + 1e-12)

def butter_lowpass_filter(sig, cutoff, fs, order=8):
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, sig)

def compute_snr(orig, recon):
    L = min(len(orig), len(recon))
    orig = orig[:L]
    recon = recon[:L]
    signal_power = np.mean(orig**2)
    noise_power = np.mean((orig - recon)**2) + 1e-12
    return 10.0 * np.log10(signal_power / noise_power)

# -----------------------
# FM: mod/demod/awgn/limiter (work at FS_MOD)
# -----------------------
def fm_modulate_baseband(message, fs_mod, fc=FC, kf=KF):
    # message sampled at fs_mod (audio resampled to fs_mod)
    int_msg = np.cumsum(message) / fs_mod
    t = np.arange(len(message)) / fs_mod
    phase = 2.0 * np.pi * (fc * t + kf * int_msg)
    return np.cos(phase).astype(np.float32)

def awgn(signal, target_snr_db):
    sig_p = np.mean(signal**2)
    target_lin = 10.0 ** (target_snr_db / 10.0)
    noise_p = sig_p / max(1e-12, target_lin)
    noise = np.sqrt(noise_p) * rng.standard_normal(size=signal.shape)
    return (signal + noise).astype(np.float32)

def limiter(sig):
    sig = np.asarray(sig, dtype=np.float32)
    peak = np.max(np.abs(sig)) + 1e-12
    s = sig / peak
    return (np.tanh(s) * peak).astype(np.float32)

def fm_demodulate_quadrature(received, fs_mod, fc=FC, kf=KF, lowpass_cut=LOWPASS_CUT_DEMOD):
    # analytic signal
    z = hilbert(received)
    prod = z[1:] * np.conj(z[:-1])
    dphase = np.angle(prod)
    inst_freq = dphase * fs_mod / (2.0 * np.pi)
    demod = (inst_freq - fc) / kf
    demod = np.concatenate([demod, [demod[-1]]])
    # lowpass at audio BW
    demod = butter_lowpass_filter(demod, lowpass_cut, fs_mod)
    return demod.astype(np.float32)

# -----------------------
# Dataset helpers
# -----------------------
def make_long_audio_from_folder(folder):
    files = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".wav")])
    if len(files) == 0:
        raise RuntimeError("No WAV files found in data_wavs/")
    parts = []
    for f in files:
        y, sr = librosa.load(f, sr=SR, mono=True)
        parts.append(y.astype(np.float32))
    audio = np.concatenate(parts).astype(np.float32)
    audio = normalize(audio)
    return audio

def frames_from_signal(sig, frame_len=FRAME, hop=HOP):
    frames = librosa.util.frame(sig, frame_length=frame_len, hop_length=hop).T
    return frames

def overlap_add(frames, hop):
    frame_len = frames.shape[1]
    n_frames = frames.shape[0]
    out_len = (n_frames - 1) * hop + frame_len
    out = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    win = get_window("hann", frame_len, fftbins=False).astype(np.float32)
    for i in range(n_frames):
        start = i * hop
        out[start:start+frame_len] += frames[i] * win
        wsum[start:start+frame_len] += win**2
    nonzero = wsum > 1e-8
    out[nonzero] /= wsum[nonzero]
    nonzero_indices = np.where(nonzero)[0]
    if nonzero_indices.size > 0:
        first, last = nonzero_indices[0], nonzero_indices[-1]
        out[:first] = out[first]
        out[last+1:] = out[last]
    return out

# -----------------------
# Neural codec reconstruct helper
# -----------------------
def reconstruct_with_model(model, signal_16k):
    frames = frames_from_signal(signal_16k, FRAME, HOP)
    frames_in = frames[..., np.newaxis].astype(np.float32)
    print("Neural: number of frames to reconstruct:", frames_in.shape[0])
    preds = model.predict(frames_in, batch_size=64, verbose=1)
    preds = preds.squeeze(-1)
    recon = overlap_add(preds, HOP)
    recon = normalize(recon)
    return recon.astype(np.float32)

# -----------------------
# Plot helpers
# -----------------------
def save_waveforms_plot(orig, analog, neural, sr, fname):
    nshow = min(len(orig), sr * 3)
    times = np.arange(nshow) / sr
    plt.figure(figsize=(12,8))
    plt.subplot(3,1,1); plt.title("Original (first 3s)"); plt.plot(times, orig[:nshow], linewidth=0.6)
    plt.subplot(3,1,2); plt.title("Analog FM recovered (first 3s)"); plt.plot(times, analog[:nshow], linewidth=0.6)
    plt.subplot(3,1,3); plt.title("Neural FM recovered (first 3s)"); plt.plot(times, neural[:nshow], linewidth=0.6)
    plt.tight_layout(); plt.savefig(fname); plt.close()

def save_spectrogram(sig, sr, fname, title="Spectrogram"):
    import librosa.display
    D = librosa.amplitude_to_db(np.abs(librosa.stft(sig, n_fft=2048)), ref=np.max)
    plt.figure(figsize=(8,3)); librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz')
    plt.title(title); plt.colorbar(format="%+2.0f dB"); plt.tight_layout(); plt.savefig(fname); plt.close()

# -----------------------
# Main
# -----------------------
def main():
    print("FM pipeline (upsample-to-mod-rate) starting.")

    # 1) Build long audio
    orig_16k = make_long_audio_from_folder(DATA_DIR)
    orig_16k = butter_lowpass_filter(orig_16k, LOWPASS_CUT_AUDIO, SR)  # limit to audio BW
    orig_16k = normalize(orig_16k)
    print("Original length (s):", len(orig_16k)/SR)

    # 2) Load model
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("trained model not found: " + MODEL_PATH)
    print("Loading model:", MODEL_PATH)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 3) Neural reconstruction (at 16k) — pre-FM file
    neural_recon_16k = reconstruct_with_model(model, orig_16k)
    sf.write(os.path.join(OUT_DIR, "neural_reconstructed_pre_fm_16k.wav"), neural_recon_16k, SR)
    print("Saved neural reconstruction at 16k.")

    # 4) Resample audio to FS_MOD for modulation
    print("Resampling audio to modulation sample rate:", FS_MOD)
    orig_mod = librosa.resample(orig_16k, orig_sr=SR, target_sr=FS_MOD)
    neural_mod = librosa.resample(neural_recon_16k, orig_sr=SR, target_sr=FS_MOD)

    # ensure band-limited at FS_MOD/2
    orig_mod = butter_lowpass_filter(orig_mod, min(LOWPASS_CUT_AUDIO, FS_MOD/2*0.98), FS_MOD)
    neural_mod = butter_lowpass_filter(neural_mod, min(LOWPASS_CUT_AUDIO, FS_MOD/2*0.98), FS_MOD)

    # 5) For each SNR run analog & neural chains at FS_MOD
    results = []
    for snr in SNR_LEVELS:
        print(f"\n--- SNR = {snr} dB ---")

        # analog: orig_mod -> FM -> AWGN -> limiter -> demod -> lowpass -> resample to 16k
        fm_tx = fm_modulate_baseband(orig_mod, FS_MOD, fc=FC, kf=KF)
        fm_tx_noisy = awgn(fm_tx, snr)
        fm_tx_noisy = limiter(fm_tx_noisy)
        rec_analog_mod = fm_demodulate_quadrature(fm_tx_noisy, FS_MOD, fc=FC, kf=KF, lowpass_cut=LOWPASS_CUT_DEMOD)
        # decimate / resample -> back to 16k for eval
        rec_analog_16k = librosa.resample(rec_analog_mod, orig_sr=FS_MOD, target_sr=SR)
        rec_analog_16k = butter_lowpass_filter(rec_analog_16k, LOWPASS_CUT_DEMOD, SR)
        rec_analog_16k = normalize(rec_analog_16k)
        sf.write(os.path.join(OUT_DIR, f"recovered_analog_snr{snr}.wav"), rec_analog_16k.astype(np.float32), SR)
        print("Saved recovered_analog_snr", snr)

        # neural chain: neural_mod -> FM -> AWGN -> limiter -> demod -> resample
        fm_tx_nn = fm_modulate_baseband(neural_mod, FS_MOD, fc=FC, kf=KF)
        fm_tx_nn_noisy = awgn(fm_tx_nn, snr)
        fm_tx_nn_noisy = limiter(fm_tx_nn_noisy)
        rec_neural_mod = fm_demodulate_quadrature(fm_tx_nn_noisy, FS_MOD, fc=FC, kf=KF, lowpass_cut=LOWPASS_CUT_DEMOD)
        rec_neural_16k = librosa.resample(rec_neural_mod, orig_sr=FS_MOD, target_sr=SR)
        rec_neural_16k = butter_lowpass_filter(rec_neural_16k, LOWPASS_CUT_DEMOD, SR)
        rec_neural_16k = normalize(rec_neural_16k)
        sf.write(os.path.join(OUT_DIR, f"recovered_neural_snr{snr}.wav"), rec_neural_16k.astype(np.float32), SR)
        print("Saved recovered_neural_snr", snr)

        # SNR compute (orig_16k vs recovered)
        snr_analog = compute_snr(orig_16k, rec_analog_16k)
        snr_neural = compute_snr(orig_16k, rec_neural_16k)
        results.append((snr, snr_analog, snr_neural))
        print(f"SNRs -> analog: {snr_analog:.2f} dB, neural: {snr_neural:.2f} dB")

        # plots
        save_waveforms_plot(orig_16k, rec_analog_16k, rec_neural_16k, SR, os.path.join(OUT_DIR, f"waveforms_snr{snr}.png"))
        save_spectrogram(orig_16k, SR, os.path.join(OUT_DIR, f"spec_orig_snr{snr}.png"), title="Original")
        save_spectrogram(rec_analog_16k, SR, os.path.join(OUT_DIR, f"spec_analog_snr{snr}.png"), title=f"Analog recovered SNR{snr}")
        save_spectrogram(rec_neural_16k, SR, os.path.join(OUT_DIR, f"spec_neural_snr{snr}.png"), title=f"Neural recovered SNR{snr}")

    # 6) CSV summary
    import csv
    csv_path = os.path.join(OUT_DIR, "snr_comparison.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["channel_snr_db", "snr_analog_db", "snr_neural_db"])
        for r in results:
            writer.writerow(r)
    print("Saved SNR table to:", csv_path)
    print("Done. Outputs in", OUT_DIR)

if __name__ == "__main__":
    main()
