# DGX Spark / ASUS GX10 — Qwen3.8 llama.cpp production patch set

This repository packages the exact `llama.cpp` integration (7 SHA-256-verified patches) used by a production ASUS GX10 / NVIDIA DGX Spark configuration running Qwen3.8-Flash-Next and Qwen3.8-27B.

The goal is **reproducibility and attribution**, not to claim original authorship of all underlying work. The production tree combines public `llama.cpp` and Unsloth work with local integration, compatibility fixes, and extensions. See [`PROVENANCE.md`](PROVENANCE.md) before reusing or redistributing the patch set.

## Snapshot

- Hardware: ASUS GX10 / NVIDIA DGX Spark (GB10)
- Snapshot date: 2026-09-16
- Production integration HEAD recorded on the source machine: `be9d75bd932947c20f36acdb7668c9b330253ca7`
- Public starting base: `a9e9c3c5f` (on Unsloth's `mtp/qwen4exp-nextn` branch)
- Local base commits `66bb57b3f` / `e5196486f` are **not on any public remote** — they are now exported as patches `00a`/`00b` (see below)
- Upstream merge: `1bc7a5af0`
- Patches: 7 (2 base + 5 integration), SHA-256 verified
- Reconstruction verified 2026-09-23: public clones + these patches produce a source tree **byte-identical** to production (`tree 5a421ae1c9948aca981aa5f7db59e5d2ec0ceeae`)

> **Heads-up (why this changed):** earlier revisions of this README told users to
> `git checkout 66bb57b3f…`. That commit exists only on the author's private branch,
> so the checkout fails with `fatal: not a tree` from any public clone. Patches
> `00a`/`00b` now carry those base commits, and the sequence below works from
> public repositories only.

## Repository layout

```text
.
├── README.md
├── LICENSE
├── NOTICE.md
├── PROVENANCE.md
├── patches/
│   ├── 00a-base-qwen4exp-direct-reads-for-lazy-PLE-table.patch
│   ├── 00b-base-lazy-share-direct-read-machinery-gemma4.patch
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

Expected result: all seven files report `OK`.

## Exact production-tree lineage

The five integration patches do **not** form a single linear series on one public base.

- Patches **00a–00b** recreate the private base commits (lazy/direct-read work) on top of a public Unsloth commit.
- Patch **01–02** follow the Unsloth lineage.
- Patches **03–05** are applied after merging the newer upstream tree.

The exact reconstruction sequence used for the documented production tree is:

```bash
git clone https://github.com/unslothai/llama.cpp.git
cd llama.cpp

git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch origin
git fetch upstream

# 1. Public base commit (Unsloth mtp/qwen4exp-nextn lineage)
git checkout a9e9c3c5f

# 2. Recreate the private base commits (lazy/direct-read machinery)
git am /path/to/this-repo/patches/00a-base-qwen4exp-direct-reads-for-lazy-PLE-table.patch
git am /path/to/this-repo/patches/00b-base-lazy-share-direct-read-machinery-gemma4.patch

# 3. Hybrid/recurrent checkpoint restore
git am /path/to/this-repo/patches/01-server-fix-hybrid-recurrent-checkpoint-restore.patch

# 4. Merge the recorded upstream tree
git merge 1bc7a5af0
#    Two files conflict: src/models/qwen4exp.cpp and tools/llama-bench/llama-bench.cpp
#    (the private lazy/direct-read work vs upstream changes). Resolve them from the
#    production blob contents, e.g. after `git checkout --theirs` re-apply the
#    lazy-mode enum strings and PLE/direct-read members, then:
#      git add -A && git commit

# 5. Apply MTP shared-tensor integration
git am /path/to/this-repo/patches/02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch

# 6. Apply qwen4exp / QSA integration patches
git am /path/to/this-repo/patches/03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
git am /path/to/this-repo/patches/04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
git am /path/to/this-repo/patches/05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch

# 7. Verify the resulting source tree matches production
git rev-parse HEAD^{tree}
# expected: 5a421ae1c9948aca981aa5f7db59e5d2ec0ceeae
```

A helper script implementing the same sequence is included:

```bash
./scripts/reconstruct-production-tree.sh ../llama.cpp-gx10
```

The script intentionally stops on conflicts rather than guessing at a resolution.

### Important note about commit hashes

The production integration branch recorded `be9d75bd932947c20f36acdb7668c9b330253ca7` as its final commit. A reconstruction can produce the same source tree while receiving different Git commit IDs if merge/committer metadata differ. Use the recorded commits, patch hashes, file content, and build behavior as the reproducibility checks; do not treat a different locally generated merge hash alone as evidence of a different source tree.

## Blob-level verification for patch 01

Applying patch 01 on top of base `a9e9c3c5f` + patches 00a/00b (which recreate the private base commits) reproduces byte-for-byte the production `tools/server/server-context.cpp` blob recorded as:

```text
ab31907c0…
```

This verifies that first source step against the production tree. It does **not** imply patch 01 alone reproduces the full production build.

## What the patches cover

0a/0b. **Lazy direct-read base work** — the private base commits (`e5196486f`, `66bb57b3f`): qwen4exp PLE-table direct reads (`--lazy-mode on-direct`) and the shared lazy-reader machinery. Not present on any public remote; exported here so the tree is reconstructible from public clones alone.
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

The Phase 2 technical write-up used alongside this patch set is posted in [`Single Spark Qwen3.8 Flash-Next + Qwen3.8 27B Left/Right Brain Hermes MoA (35-39 toks/s)
`](https://forums.developer.nvidia.com/t/single-spark-qwen3-8-flash-next-qwen3-8-27b-left-right-brain-hermes-moa-35-39-toks-s/383579).

## License and third-party work

The underlying `llama.cpp` project and the Unsloth `llama.cpp` fork are MIT-licensed. This repository preserves that MIT notice and separately documents contributor/source provenance in [`PROVENANCE.md`](PROVENANCE.md) and [`NOTICE.md`](NOTICE.md).

The patch `From:` headers reflect the local production integration branch. They should **not** be interpreted as sole authorship of the underlying ideas or code.
