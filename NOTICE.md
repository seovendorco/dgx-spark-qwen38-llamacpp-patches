# Third-party notices

This repository distributes a patch series against [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp) and the [`unslothai/llama.cpp`](https://github.com/unslothai/llama.cpp) fork.

Both source repositories are distributed under the MIT license. The repository-level `LICENSE` preserves the `llama.cpp` MIT copyright notice and adds a notice for original SV packaging/documentation and local contributions.

The patches are an integration artifact: several contain or adapt work originally authored in public upstream branches, pull requests, issues, or discussions. The patch `From:` headers identify the local integration commits and are **not** a complete authorship record.

See [`PROVENANCE.md`](PROVENANCE.md) for the detailed credit trail, including:

- spiritbuun / `buun-llama-cpp` PR #26
- Gaston Parravicini / Tongas and llama.cpp issue #22384
- Daniel Han / Unsloth and related Qwen3.8 MTP work
- Aman Gupta / `am17an` and upstream qwen4exp/sparse-FA work
- `lukolszewski` and `logari81` for public qwen4exp sparse-attention analysis

No ownership claim over those upstream contributions is intended.
