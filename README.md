# Qwen3 TTS API

OpenAI-compatible API servers for [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), enabling self-hosted text-to-speech via the standard `/v1/audio/speech` endpoint.

## Implementations

| Language | Directory | Status |
|----------|-----------|--------|
| Python | [python/](python/) | Available |
| Rust | `rust/` (planned) | Coming soon |

The **Python** implementation is a FastAPI server built on the `qwen-tts` Python package. See [python/README.md](python/README.md) for setup, Docker images, API reference, and usage examples.

The **Rust** implementation will be built on the [qwen3_tts](https://crates.io/crates/qwen3_tts) Rust crate. Stay tuned.

## Why

The purpose of these API servers is to provide self-hosted, free backend TTS services for projects such as:

- [Clawdbot](https://github.com/clawdbot/clawdbot) -- AI agent
- [EchoKit](https://github.com/second-state/echokit_server) -- Voice AI device
- [Olares](https://github.com/beclab/Olares) -- Personal AI cloud OS
- [GaiaNet](https://github.com/GaiaNet-AI/gaianet-node) -- Incentivized AI agent network and marketplace

Any application that speaks the OpenAI TTS API can swap in this server as a drop-in replacement.

## License

See [LICENSE](LICENSE).
