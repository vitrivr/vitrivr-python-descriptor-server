import base64
import io
import traceback
import threading
import tempfile
import shutil
import av
import torch
import whisper
import soundfile as sf
import numpy as np
import librosa
from urllib.parse import unquote
from flask import request, jsonify
from apiflask import APIBlueprint
from faster_whisper import WhisperModel

asr_whisper = APIBlueprint('asr_whisper', __name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
whisper_model = None
video_model = None
video_model_lock = threading.Lock()

# global lock to make whisper thread-safe
whisper_lock = threading.Lock()

@torch.inference_mode()
@asr_whisper.post("/extract/asr")
@asr_whisper.doc(summary="Extracts text from a base64-encoded WAV audio file using OpenAI Whisper ASR.")
def extract_asr():
    global whisper_model
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
            if whisper_model is None:
                whisper_model = whisper.load_model("small", device=device)
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

def decode_video_audio(video):
    """Return mono 16 kHz samples aligned to the start of the video stream."""
    with av.open(video) as container:
        if not container.streams.audio:
            return np.empty(0, dtype=np.float32)

        stream = container.streams.video[0]
        if stream.start_time is not None:
            video_start = float(stream.start_time * stream.time_base)
        else:
            video_start = (container.start_time or 0) / av.time_base

        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        chunks = []
        cursor = 0

        def append_samples(frames):
            nonlocal cursor
            for frame in frames:
                samples = frame.to_ndarray().ravel()
                if frame.pts is not None and frame.time_base is not None:
                    start = round((float(frame.pts * frame.time_base) - video_start) * 16000)
                else:
                    start = cursor
                if start > cursor:
                    chunks.append(np.zeros(start - cursor, dtype=np.int16))
                    cursor = start
                samples = samples[max(0, cursor - start):]
                chunks.append(samples)
                cursor += len(samples)

        for frame in container.decode(audio=0):
            append_samples(resampler.resample(frame))
        append_samples(resampler.resample(None))
        return np.concatenate(chunks).astype(np.float32) / 32768 if chunks else np.empty(0, dtype=np.float32)


@asr_whisper.post("/extract/asr/video")
@asr_whisper.doc(summary="Transcribes the audio of an uploaded video and returns timestamped words.")
def extract_video_asr():
    global video_model
    try:
        with tempfile.TemporaryFile() as video:
            shutil.copyfileobj(request.stream, video)
            video.seek(0)
            waveform = decode_video_audio(video)
        if not waveform.size:
            return jsonify([])

        with video_model_lock:
            if video_model is None:
                compute_type = "float16" if device.type == "cuda" else "int8"
                video_model = WhisperModel("small", device=device.type, compute_type=compute_type)
            segments, _ = video_model.transcribe(
                waveform, beam_size=1, vad_filter=True, word_timestamps=True
            )
            words = [
                {"midpointNs": round((word.start + word.end) * 500_000_000), "text": word.word}
                for segment in segments
                for word in segment.words
            ]
        return jsonify(words)
    except (av.error.InvalidDataError, IndexError):
        return jsonify({"error": "Expected a readable video file"}), 400
    except Exception:
        traceback.print_exc()
        return jsonify({"error": "Video transcription failed"}), 500
