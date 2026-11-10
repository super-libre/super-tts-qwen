# Super Voice — Qwen TTS backend

[![coverage](https://img.shields.io/endpoint?url=https://super-libre.github.io/super-voice-qwen-tts/coverage.json)](https://super-libre.github.io/super-voice-qwen-tts/)

Qwen's [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) text-to-speech models
as a subprocess backend for [Super Voice](https://github.com/jorge-menjivar/super-voice).
Ten languages at 24 kHz, with nine preset voices, a voice described in words,
or a voice cloned from a clip, on Linux and macOS, on any GPU that CUDA, ROCm,
Vulkan or Metal can drive.

## What it is

The Super Voice daemon does not compile model inference in-tree. It discovers
backends on disk and drives each one over a `/v1` HTTP contract on a Unix
socket. This is one such backend: a single Rust binary that runs the models
through [Burn](https://github.com/tracel-ai/burn) and answers
`POST /v1/synthesize`.

## License

GPL-3.0-only. See `LICENSE`, and `NOTICE` for the third-party work this
derives from and the Apache-2.0 weights the daemon fetches at runtime.
