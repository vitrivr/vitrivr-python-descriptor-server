import base64
import io
import traceback
import threading
import torch
import whisper
import soundfile as sf
import numpy as np
import librosa
from urllib.parse import unquote
from flask import request, jsonify
from apiflask import APIBlueprint

asr_whisper = APIBlueprint('asr_whisper', __name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = whisper.load_model("small", device=device)

# global lock to make whisper thread-safe
whisper_lock = threading.Lock()

@torch.inference_mode()
@asr_whisper.post("/extract/asr")
@asr_whisper.doc(summary="Extracts text from a base64-encoded WAV audio file using OpenAI Whisper ASR.")
def extract_asr():
    raw_data = request.form.get("data", "")
    data = unquote(raw_data)

    if not data or "base64," not in data:
        return jsonify({"error": "Missing or invalid base64 audio data"}), 400

    try:
        header, encoded = data.split("base64,", 1)
        audio_bytes = base64.b64decode(encoded)

        audio_file = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_file)

        waveform = np.asarray(waveform)

        # mono
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=1)
        elif waveform.ndim != 1:
            raise ValueError(f"Unexpected audio shape: {waveform.shape}")

        # float32
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        # resample to 16kHz
        if sample_rate != 16000:
            waveform = librosa.resample(
                waveform,
                orig_sr=sample_rate,
                target_sr=16000,
            )

        fp16 = (device.type == "cuda")

        #  one thread calls Whisper at a time
        with whisper_lock:
            result = whisper_model.transcribe(waveform, fp16=fp16)

        text = result.get("text", "").strip()

        return jsonify({
            "text": text,
            "language": result.get("language"),
            "segments": result.get("segments"),
        })

    except Exception as e:
        print("ERROR in /extract/asr:")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
