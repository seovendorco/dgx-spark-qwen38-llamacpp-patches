# DGX Spark / ASUS GX10 — Qwen3.8 llama.cpp production patch set

This repository packages the exact five-patch `llama.cpp` integration used by a production ASUS GX10 / NVIDIA DGX Spark configuration running Qwen3.8-Flash-Next and Qwen3.8-27B.

The goal is **reproducibility and attribution**, not to claim original authorship of all underlying work. The production tree combines public `llama.cpp` and Unsloth work with local integration, compatibility fixes, and extensions. See [`PROVENANCE.md`](PROVENANCE.md) before reusing or redistributing the patch set.

## Snapshot

- Hardware: ASUS GX10 / NVIDIA DGX Spark (GB10)
- Snapshot date: 2026-09-16
- Production integration HEAD recorded on the source machine: `be9d75bd932947c20f36acdb7668c9b330253ca7`
- Exact starting base: `66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3`
- Unsloth MTP/on-direct lineage merge: `ccc827348`
- Upstream merge: `1bc7a5af0`
- Patches: 5, SHA-256 verified

## Repository layout

```text
.
├── README.md
├── LICENSE
├── NOTICE.md
├── PROVENANCE.md
├── patches/
│   ├── 01-server-fix-hybrid-recurrent-checkpoint-restore.patch
│   ├── 02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch
│   ├── 03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
│   ├── 04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
│   ├── 05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch
│   └── PATCHES-SHA256.txt
├── scripts/
│   ├── verify-patches.sh
│   └── reconstruct-production-tree.sh
└── docs/
    ├── BUILD.md
    └── NVIDIA-FORUM-PHASE2.md
```

## Verify the patch files

Before applying anything:

```bash
./scripts/verify-patches.sh
```

Expected result: all five files report `OK`.

## Exact production-tree lineage

The five patches do **not** form a single linear series on one public base.

- Patches **01–02** are on the Unsloth lineage.
- Patches **03–05** are applied after merging the newer upstream tree.

The exact reconstruction sequence used for the documented production tree is:

```bash
git clone https://github.com/unslothai/llama.cpp.git
cd llama.cpp

git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch origin
git fetch upstream

# 1. Exact Unsloth base
git checkout 66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3

# 2. Hybrid/recurrent checkpoint restore
git am /path/to/this-repo/patches/01-server-fix-hybrid-recurrent-checkpoint-restore.patch

# 3. Bring in Unsloth's MTP/on-direct lineage
git merge ccc827348

# 4. Apply MTP shared-tensor integration
git am /path/to/this-repo/patches/02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch

# 5. Merge the recorded upstream tree
git merge 1bc7a5af0

# 6. Apply qwen4exp / QSA integration patches
git am /path/to/this-repo/patches/03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
git am /path/to/this-repo/patches/04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
git am /path/to/this-repo/patches/05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch
```

A helper script implementing the same sequence is included:

```bash
./scripts/reconstruct-production-tree.sh ../llama.cpp-gx10
```

The script intentionally stops on conflicts rather than guessing at a resolution.

### Important note about commit hashes

The production integration branch recorded `be9d75bd932947c20f36acdb7668c9b330253ca7` as its final commit. A reconstruction can produce the same source tree while receiving different Git commit IDs if merge/committer metadata differ. Use the recorded commits, patch hashes, file content, and build behavior as the reproducibility checks; do not treat a different locally generated merge hash alone as evidence of a different source tree.

## Blob-level verification for patch 01

Applying patch 01 to base commit `66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3` reproduces byte-for-byte the production `tools/server/server-context.cpp` blob recorded as:

```text
ab31907c0…
```

This verifies that first source step against the production tree. It does **not** imply patch 01 alone reproduces the full production build.

## What the patches cover

1. **Hybrid/recurrent checkpoint restore** — local adaptation of the public recurrent/hybrid checkpoint-search fix.
2. **MTP shared-tensor borrowing** — Unsloth-derived MTP integration with local conflict resolution.
3. **MTP `hc_head_norm` layout compatibility** — local compatibility follow-up to an upstream qwen4exp norm-layout change.
4. **qwen4exp HC ops** — production integration of upstream HC-op work.
5. **CUDA sparse attention / QSA integration** — local qwen4exp integration and extension on top of public sparse-FA work.

For author-by-author credit and source links, see [`PROVENANCE.md`](PROVENANCE.md).

## Build

The production build used:

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CXX_FLAGS="-march=armv9-a+sve2+i8mm"

cmake --build build -j$(nproc)
```

See [`docs/BUILD.md`](docs/BUILD.md) for the reconstruction notes and the simplified upstream-only variant.

## Simplified upstream-only variant

If you do **not** need the Unsloth `--lazy-mode on-direct` lineage, patches 03–05 can be applied on top of upstream `1bc7a5af0`.

That is useful for testing the later qwen4exp/QSA work, but it is **not** the exact production source lineage described above.

## NVIDIA forum write-up

The Phase 2 technical write-up used alongside this patch set is archived in [`docs/NVIDIA-FORUM-PHASE2.md`](docs/NVIDIA-FORUM-PHASE2.md).

## License and third-party work

The underlying `llama.cpp` project and the Unsloth `llama.cpp` fork are MIT-licensed. This repository preserves that MIT notice and separately documents contributor/source provenance in [`PROVENANCE.md`](PROVENANCE.md) and [`NOTICE.md`](NOTICE.md).

The patch `From:` headers reflect the local production integration branch. They should **not** be interpreted as sole authorship of the underlying ideas or code.
