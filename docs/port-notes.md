# Notes on the backend

The measurements and findings behind choices in the code. The code's comments
say what it does and why; the numbers, and how they were found, are here.

## Precision

The talker computes in bf16 on CUDA and in f16 on every other GPU
(`model::talker_dtype`).

- **bf16 outside CUDA.** CubeCL cannot compile it anywhere else, whatever the
  device reports.
  - On Vulkan, the SPIR-V extension for bf16 allows no arithmetic on it, yet
    CubeCL emits some, and NVIDIA's driver crashes compiling it.
  - On ROCm, CubeCL compiles through LLVM, and its lowering has no bf16 type.
    An RDNA1 card that reports bf16 failed every kernel with "Type cube.bf16
    does not have a conversion to LLVM type implemented". Nothing in the
    lowering depends on the card.
  - Metal's backend reports no bf16.
- **f16 holds the talker.** The largest value of a 1.7B prefill, the product
  inside the MLP of its third layer, is 9.6e3, against the 6.5e4 f16 reaches.
  On Vulkan the speech ends where it should.

## Warm-up

The load's warm-up generates at five prompt lengths, then, on a cold cache,
one long generation (`model::WARM_UP_LENGTHS`, `model::DEEP_WARM_UP_FRAMES`).

- **Why a ladder.** With a single 300-character warm-up on a cold cache,
  requests of 220 and 300 characters were fast. Requests of 140, 380 and 430
  characters each stalled for seconds on their own first use. One of them
  generated at 2.7 frames per second for a whole utterance, ten seconds of
  audio taking forty-seven to produce.
- **Why not a word.** A warm-up on `"Hello."` tuned a shape no real request
  has. The first real request then stalled 35 seconds before its first sound,
  against 0.4 seconds once its shape was known.
- **Why the long pass.** Before it, the first long request reached the first
  doubling of the talker's key/value cache itself. One ran at 2.6 frames per
  second for a whole utterance, ten seconds of audio taking fifty-one to
  produce. Its time to first audio looked healthy, because the cost was spread
  through the generation.
- **The threshold for the long pass.** The ladder is the probe. On the RTX 3090
  with 0.6B CustomVoice, measured 2026-10-16, it took 2.3–2.8 s against a warm
  cache. Against an empty one it took 360–395 s on CUDA and 97–100 s on Vulkan.
  `model::DEEP_WARM_UP_THRESHOLD`, 10 s, sits between the two.
- **A cold first load.** Over six minutes on CUDA on the RTX 3090.

## Kernel cache and load progress

`model::WARM_UP_CACHE_ENTRIES` is what a first load's warm-up writes to the
kernel cache from empty, tuning results and compiled kernels both:

| Accelerator | Entries | Measured with |
|---|---|---|
| CUDA | 2615 | 1.7B CustomVoice, RTX 3090 (0.6B writes as many: 2614 on 2026-10-16) |
| Vulkan | 1409 | 1.7B CustomVoice, RTX 3090 (0.6B: 1413 on 2026-10-16) |
| ROCm | 1226 | 0.6B CustomVoice, AMD BC-250 (gfx1013) |
| Metal | 822 | 0.6B CustomVoice, Apple M2 |

The BC-250 has no matrix units; a card that has them tunes more candidates and
writes more. The generic `wgpu` build is taken to be Vulkan, unmeasured.

## Streaming decode

A chunk is decoded behind `model::STREAM_CONTEXT_FRAMES` of history, 150
frames, out of the 580 the 12 Hz codec states as exact.

- **The measurement.** Run with `a_streamed_decode_matches_a_whole_decode`, on
  2026-09-21, on 239 frames of the 1.7B Base checkpoint, against a whole decode
  of the same codes:
  - behind 73 frames, the stream matched it to 37 dB;
  - behind 150, to 65 dB.

  60 dB is where the decoder's own arithmetic noise sits: its f32 matmuls run on
  tensor cores at TF32, and a whole decode is 60 dB from the fp32 reference for
  that alone. 150 frames put the seams under that noise, at 1.8x the decode work
  of 73.
- **History.** The stream first carried a hand-picked 50 frames, then the
  decoder's own 73. That figure counted one attention window where the
  decoder's eight compound. Both left every chunk past the first without most
  of its history. The decode window was 98 frames before it became 175.

## Memory

- **A request's working memory.** About 0.6 GB for the 0.6B model on the RTX
  3090. A request grows it back and keeps it for the next, which is why
  `QwenTts::release_memory` runs only once the model has been idle for a
  while.

## Cloned voices

- **The first registration of a clip length.** It tunes the encoders for that
  length. A twenty-second clip held the request open for the better part of a
  minute. On 2026-10-16 on the RTX 3090 on CUDA, with 0.6B Base, a kernel
  build's invented five-second clip took 44.8 s from an empty cache. A real
  six-second clip took 8.7 s after the build.

## ROCm logging

- **pliron at `info`.** CubeCL's ROCm compiler logs its whole IR after every
  pass. One cold load wrote a 10 GB log in about seven minutes on an AMD card,
  which is why `main` filters it to warnings.

## Burn and CubeCL pins

- **Dense convolutions as a matmul.** On a GPU without matrix units, running
  the codec vocoder's dense convolutions as a matmul over their columns made
  them 13 times faster on the AMD BC-250.

## History

- **The voice-design option.** The backend used to declare two options, and
  the description filled in for any request that named no voice. That was right
  while the option was the only way to choose a voice. Once the designs became
  voices, it was wrong: the daemon sends no voice when the user has stored no
  preference, so a typed description overrode the "· default" the picker was
  showing.
- **The cloned-voice body limit.** It was first sized to the manifest's budget.
  A budget of 15 seconds put 960 KB through it. Raising the budget to 30 put
  1.92 MB through a two-mebibyte limit, eight percent short of a bare `413`.
  It is now sized to the most the daemon can ever send.
- **CUDA release assets.** Twelve builds, one per compute capability, became
  two, one per CUDA major version, once CubeCL compiled the kernels at runtime.

## Building the kernels before the weights arrive

`POST /v1/prepare` builds the kernels from the files a model lists as its
`prepare_files`, by running the load's warm-up on a model whose weights are all
zeros (`QwenTts::build_kernels`).

Measured on 2026-10-16 on the RTX 3090 with the 0.6B CustomVoice model. Each
run was in its own process from an empty kernel cache, with the driver's own
kernel and shader caches turned off (`CUDA_CACHE_DISABLE=1`,
`__GL_SHADER_DISK_CACHE=0`), and driven the way the daemon drives it. Memory
is the process's, from `nvidia-smi`.

| | CUDA | Vulkan |
|---|---|---|
| Prepare | 364–399 s | 101–102 s |
| Load after it | 6.9 s, phase `loading` | 4.3 s, phase `loading` |
| Load sent 20 s into it, ready at | 368 s | 105 s |
| Plain first load, for comparison | 371 s | 108 s |
| Peak during the prepare | 4.0 GB | 3.4 GB |
| Left when the prepare hands over | 0.5 GB | 0.1 GB |
| Peak of the load after it | 2.9 GB | 2.5 GB |
| Peak of a plain first load | 5.0 GB | 4.4 GB |
| Longest stretch without progress | 4.8 s | 1.5 s |

- **The two prepares on CUDA.** The 399-second one shared the machine with a
  build in another repository for part of its run. The other ended 364
  seconds in, when the load sent during it began mapping weights.
- **What the load is left with.** After a prepare, the load's warm-up wrote 6
  kernel-cache entries on CUDA and 4 on Vulkan, and took 2.3 to 2.8 seconds.
  The prepare wrote 2666 to 2672 on CUDA and 1454 to 1456 on Vulkan, against
  2614 and 1413 for a plain first load, so one estimate serves both.
- **Total time.** A prepare does not shorten a first load done in one go. The
  gain is that the build no longer waits for the weights. The peak also drops
  by about a gigabyte, because the zeroed model's memory goes back to the
  device before the weights are mapped.
- **Progress during an overlap.** A load sent during the prepare kept its
  progress and reported one `initial_setup`: `building_kernels`, then
  `loading_weights` and `warming_up`, with no blank step at the handover.
- **Long requests.** After a prepare, a 452-character request gave 32 seconds
  of speech in 5.5 seconds on CUDA and 7.7 on Vulkan, as after a plain first
  load. So the doubling of the key/value cache was built too.
- **The end of speech.** The zeroed talker's draws are uniform, so it would
  sample the end of speech about once in 2049 frames and stop the long pass
  early in about one build in six. It is never sampled in a build, and every
  build's long pass ran to its 376 frames.
- **Base.** A prepare of the 0.6B Base model took 377 seconds on CUDA, and its
  load 6.8. The build runs the speaker encoder on the invented voice, so the
  load's warm-up derived that voice's embedding in 29 ms. A zeroed model writes
  nothing to the voice cache.
- **After a restart.** A second process over the same cache answered
  `prepared` at once, from the marker.
- **Shutdown.** None of the runs panicked at exit after `SIGTERM`. The backend
  installs no handler for it, so the process ends before anything is dropped.
- **The 1.7B models.** They list `prepare_files` too, but none was measured.
