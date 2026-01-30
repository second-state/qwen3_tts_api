"""FastAPI server implementing OpenAI-compatible audio APIs with Qwen3-TTS/ASR."""

import io
import logging
import os
import subprocess
import tempfile
import threading
from contextlib import asynccontextmanager
from enum import Enum
from typing import AsyncGenerator

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel, Field
from qwen_asr import Qwen3ASRModel
from qwen_tts import Qwen3TTSModel

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice mapping: OpenAI voice names -> Qwen3-TTS speaker names
# Qwen speakers: Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan, Aiden,
#                Ono_Anna, Sohee
# ---------------------------------------------------------------------------
VOICE_MAP: dict[str, str] = {
    "alloy": "Vivian",
    "ash": "Serena",
    "ballad": "Uncle_Fu",
    "coral": "Dylan",
    "echo": "Eric",
    "fable": "Ryan",
    "onyx": "Aiden",
    "nova": "Ono_Anna",
    "sage": "Sohee",
    "shimmer": "Vivian",
    "verse": "Ryan",
    "marin": "Serena",
    "cedar": "Aiden",
}

# Valid Qwen3-TTS speaker names (accepted directly in the voice field)
QWEN_SPEAKERS: set[str] = {
    "Vivian",
    "Serena",
    "Uncle_Fu",
    "Dylan",
    "Eric",
    "Ryan",
    "Aiden",
    "Ono_Anna",
    "Sohee",
}


class ResponseFormat(str, Enum):
    """Supported audio output formats."""

    mp3 = "mp3"
    opus = "opus"
    aac = "aac"
    flac = "flac"
    wav = "wav"
    pcm = "pcm"


class SpeechRequest(BaseModel):
    """Request body for POST /v1/audio/speech (OpenAI-compatible)."""

    model: str
    input: str = Field(..., max_length=4096)
    voice: str = "alloy"
    response_format: ResponseFormat = ResponseFormat.mp3
    speed: float = Field(default=1.0, ge=0.25, le=4.0)
    language: str = "Auto"
    instructions: str | None = None
    audio_sample: str | None = None
    audio_sample_text: str | None = None


# ---------------------------------------------------------------------------
# Audio encoding helpers
# ---------------------------------------------------------------------------

CONTENT_TYPES: dict[ResponseFormat, str] = {
    ResponseFormat.mp3: "audio/mpeg",
    ResponseFormat.opus: "audio/opus",
    ResponseFormat.aac: "audio/aac",
    ResponseFormat.flac: "audio/flac",
    ResponseFormat.wav: "audio/wav",
    ResponseFormat.pcm: "audio/pcm",
}


def _encode_wav(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    return buf.getvalue()


def _encode_flac(audio: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="FLAC")
    return buf.getvalue()


def _encode_pcm(audio: np.ndarray) -> bytes:
    """Return raw 16-bit signed little-endian PCM."""
    int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    return int16.tobytes()


def _encode_with_ffmpeg(
    audio: np.ndarray,
    sample_rate: int,
    fmt: ResponseFormat,
) -> bytes:
    """Encode audio via ffmpeg for mp3/opus/aac."""
    wav_bytes = _encode_wav(audio, sample_rate)

    codec_map = {
        ResponseFormat.mp3: ("libmp3lame", "mp3"),
        ResponseFormat.opus: ("libopus", "ogg"),
        ResponseFormat.aac: ("aac", "adts"),
    }
    codec, container = codec_map[fmt]

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            "pipe:0",
            "-acodec",
            codec,
            "-f",
            container,
            "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace")
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg encoding failed: {stderr}",
        )
    return result.stdout


def encode_audio(
    audio: np.ndarray,
    sample_rate: int,
    fmt: ResponseFormat,
) -> bytes:
    """Encode a numpy audio array to the requested format."""
    if fmt == ResponseFormat.wav:
        return _encode_wav(audio, sample_rate)
    if fmt == ResponseFormat.flac:
        return _encode_flac(audio, sample_rate)
    if fmt == ResponseFormat.pcm:
        return _encode_pcm(audio)
    return _encode_with_ffmpeg(audio, sample_rate, fmt)


# ---------------------------------------------------------------------------
# Speed adjustment
# ---------------------------------------------------------------------------


def apply_speed(audio: np.ndarray, speed: float) -> np.ndarray:
    """Change playback speed by resampling."""
    if speed == 1.0:
        return audio
    from scipy.signal import resample

    new_length = int(len(audio) / speed)
    if new_length == 0:
        return audio
    return resample(audio, new_length).astype(np.float32)


# ---------------------------------------------------------------------------
# Resolve voice name
# ---------------------------------------------------------------------------


def resolve_voice(voice: str) -> str:
    """Map an OpenAI voice name or Qwen speaker name to a Qwen speaker."""
    if voice in QWEN_SPEAKERS:
        return voice
    mapped = VOICE_MAP.get(voice.lower())
    if mapped is not None:
        return mapped
    raise HTTPException(
        status_code=400,
        detail=(
            f"Unknown voice '{voice}'. "
            f"Supported OpenAI voices: {sorted(VOICE_MAP.keys())}. "
            f"Supported Qwen speakers: {sorted(QWEN_SPEAKERS)}."
        ),
    )


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

_inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load the Qwen3-TTS and Qwen3-ASR models on startup."""
    model_path = os.environ.get("TTS_CUSTOMVOICE_MODEL_PATH", "")
    base_model_path = os.environ.get("TTS_BASE_MODEL_PATH", "")
    asr_model_path = os.environ.get("ASR_MODEL_PATH", "")
    device = os.environ.get("QWEN_TTS_DEVICE", "cuda:0")
    dtype_name = os.environ.get("QWEN_TTS_DTYPE", "bfloat16")
    dtype = getattr(torch, dtype_name, torch.bfloat16)
    attn_impl = os.environ.get("QWEN_TTS_ATTN", "flash_attention_2")

    if not model_path and not base_model_path and not asr_model_path:
        raise RuntimeError(
            "At least one of TTS_CUSTOMVOICE_MODEL_PATH, TTS_BASE_MODEL_PATH, "
            "or ASR_MODEL_PATH must be set."
        )

    if attn_impl == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError:
            logger.warning(
                "flash-attn is not installed, disabling flash_attention_2. "
                "Install with: pip install -U flash-attn --no-build-isolation"
            )
            attn_impl = ""

    kwargs: dict[str, object] = {
        "device_map": device,
        "dtype": dtype,
    }
    if attn_impl:
        kwargs["attn_implementation"] = attn_impl

    app.state.model = None
    if model_path:
        logger.info(
            "Loading custom-voice model %s on %s (%s, attn=%s)",
            model_path,
            device,
            dtype_name,
            attn_impl,
        )
        app.state.model = Qwen3TTSModel.from_pretrained(model_path, **kwargs)
        logger.info("Custom-voice model loaded successfully")

    app.state.base_model = None
    if base_model_path:
        logger.info(
            "Loading base model %s on %s (%s, attn=%s)",
            base_model_path,
            device,
            dtype_name,
            attn_impl,
        )
        app.state.base_model = Qwen3TTSModel.from_pretrained(base_model_path, **kwargs)
        logger.info("Base model loaded successfully")

    app.state.asr_model = None
    if asr_model_path:
        logger.info(
            "Loading ASR model %s on %s (%s, attn=%s)",
            asr_model_path,
            device,
            dtype_name,
            attn_impl,
        )
        asr_kwargs: dict[str, object] = {
            "device_map": device,
            "dtype": dtype,
            "max_inference_batch_size": 1,
            "max_new_tokens": 512,
        }
        if attn_impl:
            asr_kwargs["attn_implementation"] = attn_impl
        app.state.asr_model = Qwen3ASRModel.from_pretrained(
            asr_model_path, **asr_kwargs
        )
        logger.info("ASR model loaded successfully")

    yield


app = FastAPI(
    title="Qwen3-TTS OpenAI-Compatible API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.post("/v1/audio/speech")
async def create_speech(raw_request: Request) -> Response:
    """Generate audio from text (OpenAI-compatible endpoint).

    Accepts JSON or multipart/form-data.  Use multipart to upload
    ``audio_sample`` as a binary file for voice cloning.
    """
    req_content_type = raw_request.headers.get("content-type", "")

    if "multipart/form-data" in req_content_type:
        form = await raw_request.form()

        input_text = str(form.get("input", ""))
        if not input_text:
            raise HTTPException(status_code=422, detail="'input' field is required")
        if len(input_text) > 4096:
            raise HTTPException(
                status_code=422,
                detail="'input' exceeds 4096 characters",
            )

        voice = str(form.get("voice", "alloy"))
        try:
            fmt = ResponseFormat(str(form.get("response_format", "mp3")))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        speed = float(form.get("speed", "1.0"))
        if not 0.25 <= speed <= 4.0:
            raise HTTPException(
                status_code=422,
                detail="'speed' must be between 0.25 and 4.0",
            )
        language = str(form.get("language", "Auto"))
        instr_val = form.get("instructions")
        instructions = str(instr_val) if instr_val is not None else None
        sample_text_val = form.get("audio_sample_text")
        audio_sample_text = (
            str(sample_text_val) if sample_text_val is not None else None
        )

        audio_upload = form.get("audio_sample")
        ref_audio: tuple[np.ndarray, int] | str | None = None
        if audio_upload is not None and hasattr(audio_upload, "read"):
            audio_bytes = await audio_upload.read()
            audio_arr, audio_sr = sf.read(io.BytesIO(audio_bytes))
            ref_audio = (audio_arr.astype(np.float32), int(audio_sr))
        elif audio_upload is not None:
            ref_audio = str(audio_upload)
    else:
        request = SpeechRequest(**(await raw_request.json()))
        input_text = request.input
        voice = request.voice
        fmt = request.response_format
        speed = request.speed
        language = request.language
        instructions = request.instructions
        audio_sample_text = request.audio_sample_text
        ref_audio = request.audio_sample

    if ref_audio is not None:
        base_model: Qwen3TTSModel | None = app.state.base_model
        if base_model is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "audio_sample requires a base model. "
                    "Set TTS_BASE_MODEL_PATH to enable voice cloning."
                ),
            )
        use_icl = audio_sample_text is not None
        with _inference_lock:
            wavs, sr = base_model.generate_voice_clone(
                text=input_text,
                language=language,
                ref_audio=ref_audio,
                ref_text=audio_sample_text,
                x_vector_only_mode=not use_icl,
            )
    else:
        model: Qwen3TTSModel | None = app.state.model
        if model is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Custom-voice model is not loaded. "
                    "Set TTS_CUSTOMVOICE_MODEL_PATH to enable speaker voices, "
                    "or provide audio_sample to use voice cloning."
                ),
            )
        speaker = resolve_voice(voice)
        instruct = instructions or ""
        with _inference_lock:
            wavs, sr = model.generate_custom_voice(
                text=input_text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )

    audio: np.ndarray = wavs[0]
    audio = apply_speed(audio, speed)
    data = encode_audio(audio, sr, fmt)
    return Response(content=data, media_type=CONTENT_TYPES[fmt])


# ---------------------------------------------------------------------------
# Audio transcription helpers
# ---------------------------------------------------------------------------


def convert_audio_to_wav(audio_bytes: bytes, suffix: str = ".mp3") -> str:
    """Convert audio bytes to WAV format using ffmpeg.

    Returns the path to a temporary WAV file that the caller is responsible
    for cleaning up. If the input is already WAV format, saves it directly.
    """
    # If already WAV, save directly without conversion
    if suffix.lower() == ".wav":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
            wav_file.write(audio_bytes)
            return wav_file.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as src_file:
        src_file.write(audio_bytes)
        src_path = src_file.name

    # Create a separate temporary file path for the WAV output to avoid
    # relying on string-based suffix parsing of src_path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as wav_file:
        wav_path = wav_file.name

    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            src_path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-sample_fmt",
            "s16",
            wav_path,
        ],
        capture_output=True,
    )

    os.unlink(src_path)

    if result.returncode != 0:
        if os.path.exists(wav_path):
            os.unlink(wav_path)
        stderr = result.stderr.decode(errors="replace")
        raise HTTPException(
            status_code=500,
            detail=f"ffmpeg audio conversion failed: {stderr}",
        )

    return wav_path


# ---------------------------------------------------------------------------
# Transcription endpoint
# ---------------------------------------------------------------------------


class TranscriptionResponse(BaseModel):
    """Response for the transcription endpoint."""

    text: str


@app.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(default="qwen3-asr"),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    response_format: str = Form(default="json"),
    temperature: float = Form(default=0.0),
) -> TranscriptionResponse | Response:
    """Transcribe audio to text (OpenAI-compatible endpoint).

    Accepts multipart/form-data with an audio file.
    """
    asr_model: Qwen3ASRModel | None = app.state.asr_model
    if asr_model is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "ASR model is not loaded. Set ASR_MODEL_PATH to enable transcription."
            ),
        )

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=422, detail="Empty audio file")

    filename = file.filename or "audio.mp3"
    suffix = "." + filename.rsplit(".", 1)[-1] if "." in filename else ".mp3"

    wav_path: str | None = None
    try:
        wav_path = convert_audio_to_wav(audio_bytes, suffix=suffix)

        with _inference_lock:
            results = asr_model.transcribe(
                audio=wav_path,
                language=language,
            )

        text = results[0].text if results else ""

        if response_format == "text":
            return Response(content=text, media_type="text/plain")

        return TranscriptionResponse(text=text)

    finally:
        if wav_path and os.path.exists(wav_path):
            os.unlink(wav_path)


@app.get("/v1/models")
async def list_models() -> dict[str, object]:
    """List available models (minimal OpenAI-compatible response)."""
    models = []
    if app.state.model is not None or app.state.base_model is not None:
        models.append(
            {
                "id": "qwen3-tts",
                "object": "model",
                "owned_by": "qwen",
            }
        )
    if app.state.asr_model is not None:
        models.append(
            {
                "id": "qwen3-asr",
                "object": "model",
                "owned_by": "qwen",
            }
        )
    return {
        "object": "list",
        "data": models,
    }


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=host, port=port)
