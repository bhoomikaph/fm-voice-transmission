import os
import soundfile as sf
import librosa
import random
import shutil

SOURCE_DIR = "librispeech/test-clean"
TARGET_DIR = "data_wavs"
TARGET_MINUTES = 20  # change if you want

TARGET_SECONDS = TARGET_MINUTES * 60

# ---------------------------------------------------
# Create output folder
# ---------------------------------------------------
if not os.path.exists(TARGET_DIR):
    os.makedirs(TARGET_DIR)

# ---------------------------------------------------
# Collect all FLAC files
# ---------------------------------------------------
flac_files = []
for root, dirs, files in os.walk(SOURCE_DIR):
    for f in files:
        if f.endswith(".flac"):
            flac_files.append(os.path.join(root, f))

print(f"Found {len(flac_files)} FLAC files.")

# ---------------------------------------------------
# Shuffle files
# ---------------------------------------------------
random.shuffle(flac_files)

total_duration = 0.0
selected_files = []

# ---------------------------------------------------
# Pick files until total ≥ target duration
# ---------------------------------------------------
print("\nSelecting files...")
for f in flac_files:
    try:
        y, sr = librosa.load(f, sr=None)
        dur = len(y) / sr
        selected_files.append((f, dur))
        total_duration += dur
    except Exception as e:
        print("Error reading:", f, "->", e)
        continue
    
    if total_duration >= TARGET_SECONDS:
        break

print(f"\nSelected {len(selected_files)} files totalling: {total_duration/60:.2f} minutes")

# ---------------------------------------------------
# Convert and copy into data_wavs/
# ---------------------------------------------------
print("\nConverting to WAV 16 kHz...")

count = 1

for f, dur in selected_files:
    y, sr = librosa.load(f, sr=16000, mono=True)
    out_name = f"sample_{count:04d}.wav"
    out_path = os.path.join(TARGET_DIR, out_name)
    sf.write(out_path, y, 16000)
    count += 1

print("\nDone! WAV files saved in:", TARGET_DIR)
