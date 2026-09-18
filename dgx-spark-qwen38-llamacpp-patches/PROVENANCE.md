# Patch provenance and credits

This archive contains the exact five patch files used to reconstruct the documented GX10 / DGX Spark production llama.cpp tree. The patch `From:` fields reflect the local integration branch and should not be interpreted as sole authorship of the underlying work.

## 01 - hybrid/recurrent checkpoint restore

Local port/adaptation of the checkpoint-search fix already implemented in:

- spiritbuun/buun-llama-cpp PR #26: https://github.com/spiritbuun/buun-llama-cpp/pull/26
- ggml-org/llama.cpp issue #22384 by Tongas / Gaston Parravicini: https://github.com/ggml-org/llama.cpp/issues/22384

This patch carries the `pos_max <= pos_next` checkpoint-search fix, adapted to the newer server code in this branch. It does not include the separate 64-to-4 checkpoint-creation threshold change described in #22384.

## 02 - MTP shared-tensor borrowing

Derived from Unsloth `origin/mtp/borrow-target-tensors` commit `f36eda840`; the local production commit records 11 conflict hunks resolved during integration. Related upstream Qwen3.8 MTP work is in PR #28243 from Daniel Han / Unsloth:

https://github.com/ggml-org/llama.cpp/pull/28243

Credit: Unsloth / Daniel Han for the shared-tensor and Qwen3.8 MTP work; local work was branch integration and conflict resolution.

## 03 - MTP hc_head_norm layout compatibility

Local compatibility follow-up to upstream PR #28896 (`qwen4exp: enable rms_norm + mul fusion`) by Aman Gupta / am17an:

https://github.com/ggml-org/llama.cpp/pull/28896

The patch updates the MTP-only norm layout for the newer 2D fused-norm expectation.

## 04 - qwen4exp HC ops

The patch materially corresponds to upstream PR #28901 (`qwen4exp: add hc ops`) by Aman Gupta / am17an, merged as `37b53fd4`:

https://github.com/ggml-org/llama.cpp/pull/28901

The production tree retains a local integration commit for exact reconstruction; the underlying HC-op work is credited upstream.

## 05 - CUDA sparse attention / QSA integration

Builds on upstream CUDA/GGML sparse-FA support from PR #27970 by Aman Gupta / am17an, merged as `8e93a977`:

https://github.com/ggml-org/llama.cpp/pull/27970

Related public qwen4exp sparse-attention diagnosis and implementation work:

- lukolszewski, issue #28734: https://github.com/ggml-org/llama.cpp/issues/28734
- logari81, discussion #28695: https://github.com/ggml-org/llama.cpp/discussions/28695

The production patch adds per-query-tile union index lists with live counts, wider `DKQ=DV=256` handling, and qwen4exp QSA enablement on top of that public foundation.

## Integrity

`PATCHES-SHA256.txt` covers the five `.patch` files. `PROVENANCE.md` is explanatory metadata and is intentionally not part of the original patch hash manifest. The patch files themselves are unchanged from the verified production-reconstruction bundle.
