# Test Plan

Integration tests for the Qwen3 Audio API server. Each phase starts a server with a different model configuration and verifies that supported requests succeed and unsupported requests return HTTP 400.

## Phase 1: TTS models (both CustomVoice and Base)

Start the server with `TTS_CUSTOMVOICE_MODEL_PATH` and `TTS_BASE_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 1 | Generate English speech with Vivian (`voice: "Vivian"`) | 200 — `phase1_vivian_english.wav` |
| 2 | Generate Chinese speech with Vivian (`voice: "Vivian"`) | 200 — `phase1_vivian_chinese.wav` |
| 3 | Clone English voice using `phase1_vivian_english.wav` as `audio_sample` | 200 — `phase1_clone_english.wav` |
| 4 | Clone Chinese voice using `phase1_vivian_chinese.wav` as `audio_sample` | 200 — `phase1_clone_chinese.wav` |

Stop the server.

## Phase 2: TTS CustomVoice model only

Start the server with only `TTS_CUSTOMVOICE_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 5 | Generate English speech with Ryan (`voice: "Ryan"`) | 200 — `phase2_ryan_english.wav` |
| 6 | Generate Chinese speech with Ryan (`voice: "Ryan"`) | 200 — `phase2_ryan_chinese.wav` |
| 7 | Send request with `audio_sample` | 400 — Base model not loaded |

Stop the server.

## Phase 3: TTS Base model only

Start the server with only `TTS_BASE_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 8 | Clone voice using `phase2_ryan_english.wav` as `audio_sample` | 200 — `phase3_clone_ryan.wav` |
| 9 | Send request with `voice: "Ryan"` (no `audio_sample`) | 400 — CustomVoice model not loaded |

Stop the server.

## Phase 4: ASR model only

Start the server with only `ASR_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 10 | Transcribe `phase1_vivian_english.wav` (English) | 200 — JSON with transcribed text |
| 11 | Transcribe `phase1_vivian_chinese.wav` (Chinese) | 200 — JSON with transcribed text |
| 12 | Call TTS endpoint with `voice: "Vivian"` | 400 — TTS model not loaded |

Stop the server.

## Phase 5: TTS + ASR Round-Trip Test

Start the server with `TTS_CUSTOMVOICE_MODEL_PATH` and `ASR_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 13 | Generate English speech with Vivian | 200 — `phase5_tts_english.wav` |
| 14 | Transcribe `phase5_tts_english.wav` with ASR | 200 — Text matches input (approximately) |
| 15 | Generate Chinese speech with Vivian | 200 — `phase5_tts_chinese.wav` |
| 16 | Transcribe `phase5_tts_chinese.wav` with ASR | 200 — Text matches input (approximately) |

Stop the server.

## Generated audio files

All `.wav` files are uploaded as GitHub Actions artifacts for review.

| File | Source |
|------|--------|
| `phase1_vivian_english.wav` | Vivian, English, CustomVoice model |
| `phase1_vivian_chinese.wav` | Vivian, Chinese, CustomVoice model |
| `phase1_clone_english.wav` | Cloned from Vivian English sample, Base model |
| `phase1_clone_chinese.wav` | Cloned from Vivian Chinese sample, Base model |
| `phase2_ryan_english.wav` | Ryan, English, CustomVoice model |
| `phase2_ryan_chinese.wav` | Ryan, Chinese, CustomVoice model |
| `phase3_clone_ryan.wav` | Cloned from Ryan English sample, Base model |
| `phase5_tts_english.wav` | Vivian, English, for ASR round-trip test |
| `phase5_tts_chinese.wav` | Vivian, Chinese, for ASR round-trip test |

## Transcription results

ASR transcription results are saved as text files for review.

| File | Source |
|------|--------|
| `phase4_transcribe_english.txt` | Transcription of phase1_vivian_english.wav |
| `phase4_transcribe_chinese.txt` | Transcription of phase1_vivian_chinese.wav |
| `phase5_roundtrip_english.txt` | Round-trip: TTS input vs ASR output |
| `phase5_roundtrip_chinese.txt` | Round-trip: TTS input vs ASR output |
