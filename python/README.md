# Qwen3-TTS OpenAI-Compatible API

A FastAPI server that wraps [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) behind an OpenAI-compatible `/v1/audio/speech` endpoint. Any client that speaks the OpenAI TTS API can point at this server and get speech from Qwen3-TTS instead.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- ffmpeg (required for mp3, opus, and aac output formats)
- A CUDA GPU is strongly recommended; CPU inference works but is slow

## Download a model

Download model weights before starting the server. Pick the model that fits your hardware:

| Model | Parameters | Use case |
|-------|-----------|----------|
| `Qwen3-TTS-12Hz-0.6B-CustomVoice` | 0.6B | Lightweight, suitable for CPU |
| `Qwen3-TTS-12Hz-1.7B-CustomVoice` | 1.7B | Higher quality, recommended for GPU |

```bash
mkdir -p models

# 0.6B (smaller)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  --local-dir ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice

# 1.7B (higher quality)
huggingface-cli download Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --local-dir ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice
```

## Quickstart with Docker

**CPU:**

```bash
docker build -t qwen-tts-api .

docker run -p 8000:8000 \
  -v ./models/Qwen3-TTS-12Hz-0.6B-CustomVoice:/model \
  qwen-tts-api
```

**CUDA GPU:**

```bash
docker build -f Dockerfile.cuda -t qwen-tts-api-cuda .

docker run --gpus all -p 8000:8000 \
  -v ./models/Qwen3-TTS-12Hz-1.7B-CustomVoice:/model \
  qwen-tts-api-cuda
```

## API reference

### `POST /v1/audio/speech`

Generate speech from text. Compatible with the [OpenAI audio speech API](https://platform.openai.com/docs/api-reference/audio/createSpeech).

**Request body (JSON):**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `model` | string | yes | -- | Model identifier (accepted for compatibility; the loaded model is always used) |
| `input` | string | yes | -- | Text to synthesize (max 4096 characters) |
| `voice` | string | yes | -- | Voice name (see table below) |
| `response_format` | string | no | `mp3` | `mp3`, `opus`, `aac`, `flac`, `wav`, or `pcm` |
| `speed` | number | no | `1.0` | Playback speed, `0.25` to `4.0` |
| `language` | string | no | `Auto` | Language of the input text (`Auto`, `English`, `Chinese`, `Japanese`, `Korean`, `French`, `German`, `Spanish`, `Italian`, `Portuguese`, `Russian`) |
| `instructions` | string | no | -- | Style/emotion instruction passed to the model |

**Response:** The raw audio bytes with the appropriate `Content-Type` header.

**Example:**

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


## Running from source

### Install dependencies

```bash
uv sync
```

### Start the server

**GPU (CUDA):**

```bash
MODEL_PATH=./models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  uv run python main.py
```

**CPU:**

```bash
MODEL_PATH=./models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  QWEN_TTS_DEVICE=cpu \
  QWEN_TTS_DTYPE=float32 \
  QWEN_TTS_ATTN="" \
  uv run python main.py
```

The server listens on `http://0.0.0.0:8000` by default.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `/model` | Path to the model directory (local path or HuggingFace model ID) |
| `QWEN_TTS_DEVICE` | `cuda:0` | Torch device (`cuda:0`, `cuda:1`, `cpu`) |
| `QWEN_TTS_DTYPE` | `bfloat16` | Model precision (`bfloat16`, `float16`, `float32`) |
| `QWEN_TTS_ATTN` | `flash_attention_2` | Attention implementation (set to empty string `""` to disable) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

