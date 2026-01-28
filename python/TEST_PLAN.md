# Test Plan

Integration tests for the Qwen3-TTS API server. Each phase starts a server with a different model configuration and verifies that supported requests succeed and unsupported requests return HTTP 400.

## Phase 1: Both models loaded

Start the server with `CUSTOMVOICE_MODEL_PATH` and `BASE_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 1 | Generate English speech with Vivian (`voice: "Vivian"`) | 200 — `phase1_vivian_english.wav` |
| 2 | Generate Chinese speech with Vivian (`voice: "Vivian"`) | 200 — `phase1_vivian_chinese.wav` |
| 3 | Clone English voice using `phase1_vivian_english.wav` as `audio_sample` | 200 — `phase1_clone_english.wav` |
| 4 | Clone Chinese voice using `phase1_vivian_chinese.wav` as `audio_sample` | 200 — `phase1_clone_chinese.wav` |

Stop the server.

## Phase 2: CustomVoice model only

Start the server with only `CUSTOMVOICE_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 5 | Generate English speech with Ryan (`voice: "Ryan"`) | 200 — `phase2_ryan_english.wav` |
| 6 | Generate Chinese speech with Ryan (`voice: "Ryan"`) | 200 — `phase2_ryan_chinese.wav` |
| 7 | Send request with `audio_sample` | 400 — Base model not loaded |

Stop the server.

## Phase 3: Base model only

Start the server with only `BASE_MODEL_PATH`.

| Step | Action | Expected result |
|------|--------|-----------------|
| 8 | Clone voice using `phase2_ryan_english.wav` as `audio_sample` | 200 — `phase3_clone_ryan.wav` |
| 9 | Send request with `voice: "Ryan"` (no `audio_sample`) | 400 — CustomVoice model not loaded |

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
