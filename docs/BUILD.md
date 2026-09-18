# Build and reconstruction notes

## Exact source lineage

The production tree documented with this repository was assembled across two public lineages.

### 1. Start from the exact Unsloth base

```bash
git clone https://github.com/unslothai/llama.cpp.git
cd llama.cpp

git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch origin
git fetch upstream

git checkout 66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3
```

Base description from the source machine:

```text
lazy: share the direct-read machinery, support gemma4 PLE tables
```

### 2. Apply patch 01

```bash
git am /path/to/repo/patches/01-server-fix-hybrid-recurrent-checkpoint-restore.patch
```

### 3. Merge the recorded Unsloth MTP/on-direct lineage

```bash
git merge ccc827348
```

### 4. Apply patch 02

```bash
git am /path/to/repo/patches/02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch
```

### 5. Merge the recorded upstream tree

```bash
git merge 1bc7a5af0
```

### 6. Apply patches 03–05

```bash
git am /path/to/repo/patches/03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
git am /path/to/repo/patches/04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
git am /path/to/repo/patches/05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch
```

## Conflict handling

The patch bundle was re-tested against the recorded lineage. If Git reports a merge conflict in your environment, stop and inspect it rather than accepting an automatic resolution blindly.

The production integration preserved the MTP/on-direct behavior when combining the Unsloth and upstream trees.

Useful commands while resolving a merge:

```bash
git status
git diff --cc
```

After resolving files:

```bash
git add <resolved-files>
git commit
```

Then continue with the next recorded step.

## Build flags used on GB10

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CXX_FLAGS="-march=armv9-a+sve2+i8mm"

cmake --build build -j$(nproc)
```

## Simplified upstream-only variant

For users who do not need the Unsloth `--lazy-mode on-direct` lineage:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
git checkout 1bc7a5af0

git am /path/to/repo/patches/03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
git am /path/to/repo/patches/04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
git am /path/to/repo/patches/05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch
```

This is a legitimate test path for the later qwen4exp/QSA integration, but it is **not** the exact production tree.

## Integrity check

From this repository:

```bash
./scripts/verify-patches.sh
```

or manually:

```bash
cd patches
sha256sum -c PATCHES-SHA256.txt
```
