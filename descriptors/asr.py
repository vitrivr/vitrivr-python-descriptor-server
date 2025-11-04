import base64
import io
import gc
import torch
import whisper
import soundfile as sf
import numpy as np
from urllib.parse import unquote
from flask import request, jsonify
from apiflask import APIBlueprint

asr_whisper = APIBlueprint('asr_whisper', __name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Choose model: tiny | base | small | medium | large
whisper_model = whisper.load_model("small", device=device)

@asr_whisper.post("/extract/asr")
@asr_whisper.doc(summary="Extracts text from a base64-encoded WAV audio file using OpenAI Whisper ASR.")
def extract_asr():
    """
    ASR endpoint that receives base64 (possibly URL-encoded) WAV data,
    decodes it, runs Whisper transcription, and returns extracted text.
    """
    # Retrieve and decode 'data' field 
    raw_data = request.form.get("data", "")
    data = unquote(raw_data)  # handle URL-encoded input like data%3Aaudio%2Fwav%3Bbase64%2C...

    # print(data)
    if not data or "base64," not in data:
        return jsonify({"error": "Missing or invalid base64 audio data"}), 400

    try:
        # Decode base64 
        header, encoded = data.split("base64,", 1)
        audio_bytes = base64.b64decode(encoded)

        # Read audio bytes into float32 numpy array 
        audio_file = io.BytesIO(audio_bytes)
        waveform, sample_rate = sf.read(audio_file)

        # Convert dtype (Whisper requires float32)
        if waveform.dtype != np.float32:
            waveform = waveform.astype(np.float32)

        # Resample to 16kHz if necessary 
        if sample_rate != 16000:
            import librosa
            waveform = librosa.resample(waveform.T, orig_sr=sample_rate, target_sr=16000)
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=0)  # Convert stereo to mono

        # Transcribe with Whisper 
        result = whisper_model.transcribe(waveform)
        text = result.get("text", "").strip()

        gc.collect()
        return jsonify({
            "text": text,
            "language": result.get("language", None),
            "segments": result.get("segments", None)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
