# Qwen3-TTS OpenAI-Compatible API

A FastAPI server that wraps [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) behind an OpenAI-compatible `/v1/audio/speech` endpoint. Any client that speaks the OpenAI TTS API can point at this server and get speech from Qwen3-TTS instead.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (required for mp3, opus, and aac output formats)
- A CUDA GPU is strongly recommended; CPU inference works but is slow

## Download models

Download model weights before starting the server. There are two model families:

- **CustomVoice** — uses built-in speaker presets selected via the `voice` parameter.
- **Base** — clones any voice from a reference audio sample via the `audio_sample` parameter.

You can load one or both. At least one is required.

| Model | Parameters | Type | Use case |
|-------|-----------|------|----------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | CustomVoice | Lightweight, suitable for CPU |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | CustomVoice | Higher quality, recommended for GPU |
| `Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | Base | Voice cloning, suitable for CPU |

```bash
mkdir -p models

# CustomVoice — 0.6B (smaller)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice

# CustomVoice — 1.7B (higher quality)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice

# Base — 0.6B (voice cloning)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --local-dir ./models/Qwen3-TTS-12Hz-0.6B-Base
```

## Quickstart with Docker

**CPU:**

```bash
docker build -t qwen-tts-api .

docker run -p 8000:8000 \
  -v ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice:/customvoice-model \
  -v ./models/Qwen3-TTS-12Hz-0.6B-Base:/base-model \
  -e CUSTOMVOICE_MODEL_PATH=/customvoice-model \
  -e BASE_MODEL_PATH=/base-model \
  qwen-tts-api
```

**CUDA GPU:**

```bash
docker build -f Dockerfile.cuda -t qwen-tts-api-cuda .

docker run --gpus all -p 8000:8000 \
  -v ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice:/customvoice-model \
  -v ./models/Qwen3-TTS-12Hz-0.6B-Base:/base-model \
  -e CUSTOMVOICE_MODEL_PATH=/customvoice-model \
  -e BASE_MODEL_PATH=/base-model \
  qwen-tts-api-cuda
```

## API reference

### `POST /v1/audio/speech`

Generate speech from text. Compatible with the [OpenAI audio speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Request body (JSON):**

| Field | Type | Required | Default | Description | Requires model |
|-------|------|----------|---------|-------------|----------------|
| `model` | string | yes | -- | Model identifier (accepted for compatibility; the loaded model is always used) | -- |
| `input` | string | yes | -- | Text to synthesize (max 4096 characters) | -- |
| `voice` | string | no | `alloy` | Voice name (see table below) | CustomVoice |
| `response_format` | string | no | `mp3` | `mp3`, `opus`, `aac`, `flac`, `wav`, or `pcm` | -- |
| `speed` | number | no | `1.0` | Playback speed, `0.25` to `4.0` | -- |
| `language` | string | no | `Auto` | Language of the input text (`Auto`, `English`, `Chinese`, `Japanese`, `Korean`, `French`, `German`, `Spanish`, `Italian`, `Portuguese`, `Russian`) | -- |
| `instructions` | string | no | -- | Style/emotion instruction passed to the model | CustomVoice |
| `audio_sample` | string | no | -- | Path, URL, or base64-encoded reference audio for voice cloning | Base |
| `audio_sample_text` | string | no | -- | Transcript of the reference audio; enables in-context learning mode for higher quality cloning | Base |

> **Note:** When `audio_sample` is provided the request uses the **Base** model for voice cloning and `voice`/`instructions` are ignored. When `audio_sample` is omitted the request uses the **CustomVoice** model and requires a valid `voice`. If the required model is not loaded the server returns HTTP 400.

**Response:** The raw audio bytes with the appropriate `Content-Type` header.

**Example — predefined voice (CustomVoice model):**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": "Hello, welcome to the Qwen text-to-speech API.",
    "voice": "alloy",
    "language": "English",
    "response_format": "wav"
  }' \
  --output speech.wav
```

**Example — voice cloning (Base model):**

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-tts",
    "input": "This sentence will be spoken in the cloned voice.",
    "audio_sample": "/path/to/reference.wav",
    "audio_sample_text": "Transcript of the reference audio.",
    "language": "English",
    "response_format": "wav"
  }' \
  --output cloned.wav
```

### `GET /v1/models`

Returns the list of available models.

### `GET /health`

Returns `{"status": "ok"}` when the server is ready.

## Voices

The `voice` field accepts OpenAI voice names (mapped to Qwen3-TTS speakers) or Qwen3-TTS speaker names directly.

**OpenAI voice mapping:**

| OpenAI voice | Qwen3-TTS speaker |
|--------------|-------------------|
| `alloy` | Vivian |
| `ash` | Serena |
| `ballad` | Uncle_Fu |
| `coral` | Dylan |
| `echo` | Eric |
| `fable` | Ryan |
| `onyx` | Aiden |
| `nova` | Ono_Anna |
| `sage` | Sohee |
| `shimmer` | Vivian |
| `verse` | Ryan |
| `marin` | Serena |
| `cedar` | Aiden |

**Qwen3-TTS speakers** can also be used directly as the voice value: `Vivian`, `Serena`, `Uncle_Fu`, `Dylan`, `Eric`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`.

## Output formats

| Format | Content-Type | Requires ffmpeg |
|--------|-------------|-----------------|
| `wav` | `audio/wav` | No |
| `flac` | `audio/flac` | No |
| `pcm` | `audio/pcm` | No |
| `mp3` | `audio/mpeg` | Yes |
| `opus` | `audio/opus` | Yes |
| `aac` | `audio/aac` | Yes |


## Preparing reference audio

The `audio_sample` parameter accepts a path to a WAV file. If your source audio is in another format (mp3, m4a, ogg, etc.), convert it with ffmpeg first:

```bash
ffmpeg -i input.m4a -ac 1 -ar 24000 -sample_fmt s16 reference.wav
```

| Flag | Meaning |
|------|---------|
| `-ac 1` | Mix down to mono |
| `-ar 24000` | Resample to 24 kHz (expected by the speaker encoder) |
| `-sample_fmt s16` | 16-bit signed PCM |

This works for any input format ffmpeg supports. A few common examples:

```bash
# MP3
ffmpeg -i recording.mp3 -ac 1 -ar 24000 -sample_fmt s16 reference.wav

# OGG / Opus
ffmpeg -i recording.ogg -ac 1 -ar 24000 -sample_fmt s16 reference.wav

# FLAC
ffmpeg -i recording.flac -ac 1 -ar 24000 -sample_fmt s16 reference.wav
```

A short clip (3–10 seconds) of clear speech with minimal background noise gives the best cloning results.

## Running from source

### Install dependencies

```bash
uv sync
```

To enable flash attention on a CUDA GPU (optional, reduces GPU memory usage):

```bash
pip install -U flash-attn --no-build-isolation
```

### Start the server

At least one of `CUSTOMVOICE_MODEL_PATH` or `BASE_MODEL_PATH` must be set. Both can be loaded at the same time.

**GPU (CUDA):**

```bash
CUSTOMVOICE_MODEL_PATH=./models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  BASE_MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-Base \
  uv run python main.py
```

**CPU:**

```bash
CUSTOMVOICE_MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  BASE_MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-Base \
  QWEN_TTS_DEVICE=cpu \
  QWEN_TTS_DTYPE=float32 \
  QWEN_TTS_ATTN="" \
  uv run python main.py
```

The server listens on `http://0.0.0.0:8000` by default.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CUSTOMVOICE_MODEL_PATH` | -- | Path to a CustomVoice model directory (enables `voice`/`instructions` parameters) |
| `BASE_MODEL_PATH` | -- | Path to a Base model directory (enables `audio_sample` voice cloning) |
| `QWEN_TTS_DEVICE` | `cuda:0` | Torch device (`cuda:0`, `cuda:1`, `cpu`) |
| `QWEN_TTS_DTYPE` | `bfloat16` | Model precision (`bfloat16`, `float16`, `float32`) |
| `QWEN_TTS_ATTN` | `flash_attention_2` | Attention implementation (set to empty string `""` to disable) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

