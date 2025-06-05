import os
import subprocess
import torch
from datetime import datetime
from TTS.api import TTS
from torch.serialization import add_safe_globals
from TTS.tts.configs.xtts_config import XttsConfig
from playsound import playsound
import librosa
import numpy as np

# Register config for XTTS
add_safe_globals([XttsConfig])

# Use CPU if requested
USE_CPU = "--cpu" in os.sys.argv
DEVICE = "cpu" if USE_CPU else ("cuda" if torch.cuda.is_available() else "cpu")

# Load TTS model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
tts.to(DEVICE)

# Paths
SOX_PATH = r"C:\Program Files (x86)\sox-14-4-2\sox.exe"
temp_path = "temp_voice.wav"

def apply_sox_effects(input_file, output_file, pitch_shift=0.0, speed=1.0):
    if not os.path.isfile(input_file):
        raise FileNotFoundError(f"❌ Input file not found: {input_file}")
    effects = []
    if abs(pitch_shift) > 0.01:
        effects += ["pitch", str(pitch_shift)]
    if abs(speed - 1.0) > 0.01:
        effects += ["speed", str(speed)]
    effects += ["rate", "22050"]
    cmd = [SOX_PATH, input_file, output_file] + effects
    print(f"🛠️ Running SoX command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def extract_pitch_and_tempo(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    speed_factor = tempo / 120.0
    pitches, mags = librosa.piptrack(y=y, sr=sr)
    voiced = pitches[mags > np.median(mags)]
    avg_pitch = np.median(voiced) if voiced.size > 0 else 150.0
    pitch_shift = 12 * np.log2(avg_pitch / 150.0)
    return pitch_shift, speed_factor

def main():
    print("🚀 Starting Dynamic TTS Program")

    # Ask for language once
    language = input("🌐 Choose language code ('hi' for Hindi, 'en' for English): ").strip().lower()

    while True:
        try:
            # Step 1: Ask for reference audio
            ref_path = input("\n🎙️ Path to reference .wav (press Enter to skip or type 'exit'): ").strip()
            if ref_path.lower() == "exit":
                print("👋 Exiting program.")
                break

            if ref_path:
                if not os.path.isfile(ref_path):
                    print(f"❌ File not found: {ref_path}")
                    continue

                print("🔍 Analyzing reference audio...")
                pitch_shift, speed_factor = extract_pitch_and_tempo(ref_path)
                print(f"🎼 Detected pitch shift: {pitch_shift:.2f} semitones")
                print(f"⏱️ Detected speed factor: {speed_factor:.2f}")

                adjust = input("⚙️ Adjust pitch/speed manually? (y/n): ").strip().lower()
                if adjust == 'y':
                    try:
                        user_pitch = input(f"🎚️ Enter pitch shift (default {pitch_shift:.2f}): ").strip()
                        pitch_shift = float(user_pitch) if user_pitch else pitch_shift
                    except ValueError:
                        print("⚠️ Invalid pitch. Using detected value.")

                    try:
                        user_speed = input(f"⏩ Enter speed factor (default {speed_factor:.2f}): ").strip()
                        speed_factor = float(user_speed) if user_speed else speed_factor
                    except ValueError:
                        print("⚠️ Invalid speed. Using detected value.")
            else:
                try:
                    pitch_shift = float(input("🎚️ Enter pitch shift (e.g., -2.0 to +2.0): ").strip())
                except ValueError:
                    pitch_shift = 0.0
                try:
                    speed_factor = float(input("⏩ Enter speed factor (e.g., 1.0 = normal): ").strip())
                except ValueError:
                    speed_factor = 1.0

            # Step 2: Ask for text input
            text = input("\n📝 Enter text to synthesize (or type 'exit'): ").strip()
            if text.lower() == "exit":
                print("👋 Exiting program.")
                break

            # Step 3: Generate audio
            print("🔊 Generating speech...")
            tts.tts_to_file(
                text=text,
                speaker_wav=ref_path if ref_path else None,
                language=language,
                file_path=temp_path
            )

            # Step 4: Apply SoX effects and save with timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"output_{timestamp}.wav"

            print("🎚️ Applying pitch and speed effects...")
            apply_sox_effects(temp_path, output_path, pitch_shift, speed_factor)

            print(f"✅ Done! Audio saved to: {output_path}")
            print("▶️ Playing generated audio...")
            playsound(output_path)

        except KeyboardInterrupt:
            print("\n👋 Program interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"⚠️ Error: {e}")

if __name__ == "__main__":
    main()