# [UPDATE] ASUS GX10 / DGX Spark, 3 months later: Qwen3.8 Flash-Next + Qwen3.8-27B, dual 96K context, MTP/DFlash2, and the memory architecture that made it stable

This is a follow-up to my June thread:

**[Stable Hermes Twin: Qwen3.6-35B-A3B / Qwen3.6-27B + ComfyUI](https://forums.developer.nvidia.com/t/asus-gx10-stable-hermes-twin-qwen3-6-35a-a3b-qwen3-6-27b-comfyui/373094)**

That post documented the point where I finally had the ASUS GX10 / DGX Spark running reliably after roughly two months of BIOS, RMA, cooling, CUDA, llama.cpp, model, and Hermes work.

Three months later, the same box is running a very different system.

The goal has also changed.

In June, I was mostly trying to prove that I could keep multiple useful local models alive at the same time without the machine falling over.

Today, the goal is **production agentic work**: two substantially more capable models, both with long context, both available to Hermes, speculative decoding on each, and enough memory headroom that the system can survive real-world bursty workloads instead of only looking good in a clean benchmark.

Everything below was pulled live from the machine on **2026-09-16**.

At the time of the snapshot, the system had been up for **6 days 15 hours without a crash**.

This is the configuration I am actually running, not a synthetic one-model benchmark.

---

## 1. TL;DR — what changed since June

| Item | June 2026 | September 2026 |
| :--- | :--- | :--- |
| llama.cpp instances | 3 slots | **2 slots — fewer models, substantially more capability per slot** |
| Primary model | Qwen3.6-35B-A3B Q4_K_M | **Qwen3.8-Flash-Next UD-Q4_K_XL**, 106 GB split GGUF, MoE + recurrent/SSM hybrid |
| Dense model | Qwen3.6-27B Q6_K | **Qwen3.8-27B NVFP4-MTP-MID-HIGH**, 16.1 GB |
| Context | 128K-class setup | **96K × 2 simultaneously** |
| Speculative decoding | MTP | **MTP + ngram-mod on Flash-Next / DFlash2 + ngram-mod on 27B** |
| Attention | Standard flash attention | **QSA sparse-attention support enabled for qwen4exp** |
| Memory strategy | mlock, essentially no swap headroom | **zswap + swappiness=1 + 100 GB swap safety valve** |
| Primary-slot decode | ~60 tok/s spot measurement @128K | median **40.1**, p90 **55.9 tok/s** @96K under production load |
| 27B decode | ~25 tok/s spot measurement | median **44.1**, p90 **62.1 tok/s** @96K under production load |
| Hermes Agent | v0.14.0, **41 cron jobs identified** | **v0.21.0, 74 defined / 26 currently enabled, dual-model MoA routing** |
| Kernel / driver | 6.17.0-1014 / 595.58.03 | **6.17.0-1021-nvidia / 610.43.02 / CUDA 13.3** |

The raw-speed story is worth stating accurately.

The new Flash-Next slot is **not faster in tok/s than the much smaller June 35B-A3B configuration**. The win is that I can now keep a far more capable 106 GB hybrid model resident beside a 27B dense model and still sustain roughly 40–56 tok/s through normal production workloads.

The 27B slot *is* substantially faster than the June version: its current 44.1 tok/s median is roughly **76% above** the old ~25 tok/s spot measurement.

The June and September measurements were collected differently, so I treat that comparison as directional rather than as a controlled benchmark. The September numbers are much more rigorous; see §7.

The cron count also needs context. The June system had **41 jobs identified in the configuration I recovered from that period**. Today there are **74 defined and 26 currently enabled**. Some older jobs were consolidated into broader workflows, so the enabled count should not be interpreted as a reduction in system use.

---

## 2. Hardware / OS baseline

Hardware is mostly unchanged from the June thread:

- ASUSTeK **GX10**
- BIOS `GX10DGX.0104.2026.0326.1657`
- GB10: 10× Cortex-X925 + 10× Cortex-A725
- aarch64 with SVE2 / i8mm / BF16
- **121 GiB usable unified memory**
- 916 GB internal NVMe
- Ubuntu 24.04.4 LTS
- Kernel `6.17.0-1021-nvidia`
- NVIDIA open kernel module **610.43.02**
- CUDA UMD **13.3**
- nvcc **13.3.33** from `/usr/local/cuda-13.3`
- Kernel cmdline includes:

```text
pcie_aspm=performance pci=pcie_bus_perf
```

Cooling posture is unchanged from June: vertical orientation, good clearance, and active ambient airflow.

The RMA replacement unit from the first thread remains stable.

---

## 3. llama.cpp — exact source tree and reproducible patch sequence

A number of people asked for more detail on the llama.cpp side in the original thread, so this time I am documenting the tree down to the individual commits.

The working branch itself is local to this GX10 and is **not** published on either remote.

The complete five-patch series used by this machine is published in the companion GitHub repository:

**Repository path:** `patches/`

The archive contains:

```text
01-server-fix-hybrid-recurrent-checkpoint-restore.patch
02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch
03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch
PATCHES-SHA256.txt
PROVENANCE.md
```

All five patches were re-tested with `git apply --check` / `git am` against their recorded lineage and apply cleanly.

The archive also includes SHA-256 hashes for every patch so the files can be verified before application.

### Remotes

```text
origin    https://github.com/unslothai/llama.cpp.git
upstream  https://github.com/ggml-org/llama.cpp.git
```

### Working tree

```text
branch:    spark-qwen38-mtp-ple
HEAD:      be9d75bd932947c20f36acdb7668c9b330253ca7
describe:  b10970-22-gbe9d75bd9

merge-base with upstream/master:
           1bc7a5af0
```

Snapshot date: **2026-09-16**

The working tree was clean when the system snapshot was taken.

### Exact base commit

The reconstruction starts from:

```text
66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3
```

Unsloth master, September 10:

> `lazy: share the direct-read machinery, support gemma4 PLE tables`

There is also a useful blob-level verification point: applying patch 01 to `66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3` reproduces byte-for-byte the same `server-context.cpp` blob (`ab31907c0…`) present in the production tree used for this build.

That verifies the first step against the actual source blob used in production; it should not be interpreted as saying patch 01 alone reproduces the full production tree.

### Production integration commits

Shown below in application order, oldest to newest:

```text
fa2970869  server: fix hybrid recurrent checkpoint restore
7e5fd8901  llama: switch MTP draft borrow to unsloth's tensor-presence gate
8e0f49ff1  qwen4exp: fix MTP hc_head_norm tensor layout after upstream 2D norm change
5da03c906  ggml + qwen4exp: fused gated hc_pre and identity-comb hc_post
be9d75bd9  cuda fattn: per-query-tile sparse index lists; enable QSA sparse attention
```

These are the commit objects in the production integration branch. The `From:` field in the exported patch files is `zternal <zternal@localhost>` because the patches were exported from that local branch. **That should not be read as a claim that all underlying ideas or code originated locally.** Several patches carry, adapt, or build on work from llama.cpp, Unsloth, and community contributors. Exact provenance is documented below and in the attached `PROVENANCE.md`.

### Important: the five patches span two lineages

This is the part that is easy to miss.

The patch set is **not** a simple five-patch linear series on top of one base commit.

- Patches **01–02** belong to the Unsloth lineage.
- Patches **03–05** are applied after merging the newer upstream tree.

The exact production-tree reconstruction sequence is:

```bash
git clone https://github.com/unslothai/llama.cpp.git
cd llama.cpp

git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch origin
git fetch upstream

# 1. Exact Unsloth base
git checkout 66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3

# 2. Hybrid recurrent checkpoint restore
git am /path/to/01-server-fix-hybrid-recurrent-checkpoint-restore.patch

# 3. Bring in Unsloth's MTP-on-direct lineage
git merge ccc827348

# 4. Apply MTP draft-borrow fix
git am /path/to/02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch

# 5. Merge upstream Sep 15 tree
git merge 1bc7a5af0

# 6. Apply qwen4exp / QSA patches
git am /path/to/03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
git am /path/to/04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
git am /path/to/05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch
```

The exact production tree should end at:

```text
be9d75bd932947c20f36acdb7668c9b330253ca7
```

If a merge conflict appears while reconstructing the two-lineage tree, resolve it to preserve the production-side MTP/on-direct changes before applying the next patch, then verify that the final tree reaches the documented HEAD above. The repository patches were tested against this recorded sequence.

### Simplified upstream-only variant

If you **do not need `--lazy-mode on-direct`**, there is a legitimate shortcut:

Start from upstream `1bc7a5af0` and apply patches 03–05 only.

That reproduces the later qwen4exp/QSA work, but it is **not byte-identical to the production tree described in this post** and it does not reproduce the Unsloth MTP/on-direct lineage.

### Patch SHA-256 manifest

```text
702b240e68a72a3b81d2f11c75fae26ce1f7de8a0056cd329402acbd18d10645  01-server-fix-hybrid-recurrent-checkpoint-restore.patch
7922f8ada914ea80c188f19ee1fef68891e23731137798f644b124d8f735b595  02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch
509423f413706b4c6848d5f7cb6e14aaa4911b3c7276902f9fb862eef1dd11d9  03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch
717bd7a45e1be52a109113d74c573d7442ce2c5112685368ac818b26e8dc1a5c  04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch
63884285ec680816db9812690a0a3ae48271b7362982722d0afc3b255d6454aa  05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch
```

### Patch provenance and credits

I want to be explicit about attribution. This patch bundle exists to reproduce the exact production tree, not to claim original authorship of every change it contains.

#### Patch 01 - hybrid/recurrent checkpoint restore

`fa2970869` is a local port/adaptation of the checkpoint-search fix that had already been implemented in [`spiritbuun/buun-llama-cpp#26`](https://github.com/spiritbuun/buun-llama-cpp/pull/26) and was documented for upstream llama.cpp in [issue #22384](https://github.com/ggml-org/llama.cpp/issues/22384) by Tongas / Gaston Parravicini.

The public fix identified two issues. This production patch carries the `pos_max <= pos_next` checkpoint-search change for hybrid/recurrent models, adapted to the newer `server-context.cpp` around this branch. It does **not** claim the underlying checkpoint-search idea as original, and it does not include the separate 64-to-4 checkpoint-creation-threshold change described in #22384.

**Credit:** spiritbuun / `buun-llama-cpp` PR #26; Tongas (Gaston Parravicini) for the upstream report and analysis in #22384.

#### Patch 02 - MTP shared-tensor borrowing

`7e5fd8901` is explicitly derived from Unsloth's `origin/mtp/borrow-target-tensors` commit `f36eda840`. The production commit message records that it was cherry-picked with **11 conflict hunks resolved locally** while preserving the existing qwen4exp MTP work in this branch.

This is therefore **Unsloth-derived code with local integration work**, not an original local implementation. Related Qwen3.8 MTP work is also visible in upstream llama.cpp [PR #28243](https://github.com/ggml-org/llama.cpp/pull/28243) from Daniel Han / Unsloth.

**Credit:** Unsloth and Daniel Han for the Qwen3.8 MTP/shared-tensor work; local work here was branch integration and conflict resolution.

#### Patch 03 - MTP `hc_head_norm` layout compatibility

`8e0f49ff1` is a local compatibility fix required after upstream [PR #28896](https://github.com/ggml-org/llama.cpp/pull/28896), `qwen4exp: enable rms_norm + mul fusion`, by Aman Gupta (`am17an`), changed the relevant norm representation to the 2D layout expected by the fused path.

The production patch updates the MTP-only `nextn.hc_head_norm` tensor declaration to match that layout and marks `tok_embd` non-required for an MTP-only head. I consider this a **local follow-up to an upstream change**, not a standalone upstream invention.

**Credit:** Aman Gupta / `am17an` for upstream PR #28896 and the norm/fusion change that this compatibility patch follows.

#### Patch 04 - qwen4exp HC ops

`5da03c906` should **not** be presented as wholly original local work. The patch materially corresponds to the qwen4exp HC-op work published upstream as [PR #28901](https://github.com/ggml-org/llama.cpp/pull/28901), `qwen4exp: add hc ops`, by Aman Gupta (`am17an`), later merged as commit `37b53fd4`.

The exact production tree carries this work as a local integration commit because of the branch lineage documented above. The bundle keeps that commit so the running tree can be reconstructed exactly, but the underlying HC-op work is credited upstream.

**Credit:** Aman Gupta / `am17an`, upstream PR #28901 and merged commit `37b53fd4`.

#### Patch 05 - CUDA sparse attention / QSA integration

`be9d75bd9` is a local extension of existing sparse-flash-attention work rather than a from-scratch sparse-attention implementation. The underlying CUDA/GGML sparse-FA infrastructure was added upstream in [PR #27970](https://github.com/ggml-org/llama.cpp/pull/27970), `CUDA + ggml: add sparse-fa for DSV4/GLM`, by Aman Gupta (`am17an`), merged as `8e93a977`.

The Qwen3.8/qwen4exp long-context sparse-attention problem and enablement path were also being worked through publicly at the same time. In [issue #28734](https://github.com/ggml-org/llama.cpp/issues/28734), `lukolszewski` documented the qwen4exp context-depth slowdown and the sparse-FA enablement/head-shape path. In [discussion #28695](https://github.com/ggml-org/llama.cpp/discussions/28695), `logari81` documented sparse-FA work for qwen4exp and important bounds/compaction considerations.

The production patch adds its own integration on top of that foundation: per-query-tile union index lists with live counts, the wider `DKQ=DV=256` handling used here, and qwen4exp QSA enablement for this branch.

**Credit:** Aman Gupta / `am17an` for the upstream sparse-FA framework (#27970); `lukolszewski` for #28734; `logari81` for #28695 and the surrounding qwen4exp sparse-FA analysis. The per-query-tile integration in this production patch is a local extension of that public foundation.

The goal of keeping all five files in the repository is **exact production-tree reconstruction**. The provenance notes above are the authorship/credit record; the local Git commit authors alone are not.

### What the patches do

#### Hybrid recurrent checkpoint restore

`fa2970869`

Fixes a real failure mode in hybrid/recurrent state restoration around speculative decode checkpoints. More on this in §8.

#### MTP draft attachment

`7e5fd8901`

Switches draft borrowing to Unsloth's tensor-presence gate so the shared MTP head attaches correctly to the split Flash-Next model.

#### Qwen hybrid-path fixes

`8e0f49ff1` and `5da03c906`

These correct the MTP norm tensor layout after an upstream 2D norm change and fuse the relevant `hc_pre` / `hc_post` paths for qwen4exp.

#### QSA sparse attention

`be9d75bd9`

This enables sparse-attention support for the qwen4exp path using per-query-tile sparse index lists in the CUDA flash-attention implementation.

One caveat: **QSA was enabled on this system on September 16, and I do not yet have a controlled before/after benchmark.**

The production performance numbers in §7 are all from the post-QSA configuration, but they should not be interpreted as proof of a specific QSA speedup.

What I can say today is that the patch enables the sparse-attention path needed for the current 96K Flash-Next configuration and has remained stable under production traffic.

If I run a controlled long-context A/B later, I will add the numbers to this thread.

### Build

```bash
cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121a-real \
  -DGGML_CUDA_GRAPHS=ON \
  -DGGML_NATIVE=ON \
  -DCMAKE_CXX_FLAGS="-march=armv9-a+sve2+i8mm"

cmake --build build -j$(nproc)
```

I am deliberately using `121a-real` for the GB10.

That has worked better for this dedicated machine than relying on a virtual/PTX target and runtime JIT behavior.

`GGML_CUDA_FA_ALL_QUANTS` is now **OFF**. The June build had it enabled, but the quantized flash-attention paths I currently use do not require the blanket build option.

---

## 4. The two model slots

The new architecture deliberately uses two very different models.

They are not redundant.

One is the primary reasoning/tool-use model. The other is the dense workhorse/reference model.

### Slot A — port 10994

### Qwen3.8-Flash-Next UD-Q4_K_XL

Primary model for:

- interactive Hermes use
- difficult reasoning
- tool calling
- delegation
- MoA aggregation
- tasks where overall model capability matters more than absolute tok/s

Files:

- 106 GB, four-part main GGUF
- shared MTP head
- BF16 vision `mmproj`

Launch:

```bash
./llama.cpp/build/bin/llama-server \
  --model ~/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
  --model-draft ~/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf \
  --mmproj ~/models/Qwen3.8-Flash-Next-UD-Q4_K_XL/mmproj-BF16.gguf \
  --alias Qwen3.8-Flash-Next \
  --port 10994 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.00 \
  --presence_penalty 1.5 \
  --flash-attn on \
  --n-gpu-layers 999 \
  --lazy-mode on-direct \
  --parallel 1 \
  --ctx-size 98304 \
  --cont-batching \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --cache-ram -1 \
  --kv-unified \
  --reasoning off \
  --reasoning-budget 0 \
  --reasoning-preserve \
  --threads 8 \
  --spec-type draft-mtp,ngram-mod \
  --spec-draft-n-max 3 \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 4 \
  --spec-ngram-mod-n-max 32
```

### Slot B — port 10993

### Qwen3.8-27B NVFP4-MTP-MID-HIGH

This is the dense model.

It serves as:

- cron/workflow model
- long-context precision model
- MoA reference model
- second opinion for analytical tasks
- lower-memory fallback when the larger slot is busy

Main model size: **16.1 GB**

Drafter: **DFlash2**, approximately 1.1 GB.

Launch:

```bash
./llama.cpp/build/bin/llama-server \
  --model ~/models/Qwen3.8-27B-NVFP4-MTP-MID-HIGH/Qwen3.8-27B-NVFP4-MTP-MID-HIGH.gguf \
  --model-draft ~/models/Qwen3.8-27B-DFlash2/Qwen3.8-27B-DFlash2-Q4_K_M.gguf \
  --mmproj ~/models/Qwen3.8-27B-NVFP4-MTP-MID-HIGH/mmproj-BF16.gguf \
  --alias Qwen3.8-27B-NVFP4 \
  --port 10993 \
  --temp 0.7 \
  --top-p 0.8 \
  --top-k 20 \
  --min-p 0.00 \
  --presence_penalty 1.5 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --parallel 1 \
  --ctx-size 98304 \
  --cache-ram -1 \
  --cont-batching \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --kv-unified \
  --reasoning off \
  --reasoning-budget 0 \
  --reasoning-preserve \
  --threads 8 \
  --spec-type draft-dflash,ngram-mod \
  --spec-draft-n-max 7 \
  --spec-ngram-mod-n-match 24 \
  --spec-ngram-mod-n-min 4 \
  --spec-ngram-mod-n-max 32 \
  --jinja
```

### Model sources

These sources were recovered from the actual machine and download records rather than reconstructed from memory.

#### Qwen3.8-Flash-Next UD-Q4_K_XL

Main split GGUF:

```text
https://huggingface.co/unsloth/Qwen3.8-Flash-Next-UD-Q4_K_XL-GGUF
```

Base checkpoint:

```text
https://huggingface.co/Qwen/Qwen3.8-Flash-Next
```

#### Flash-Next shared MTP head

Repository:

```text
https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF
```

Exact file:

```text
MTP/mtp-Qwen3.8-Flash-Next-shared-Q4_K_M.gguf
```

The downloaded MTP file was verified against its Hugging Face LFS SHA-256 object ID at download time.

#### Qwen3.8-27B NVFP4-MTP-MID-HIGH

```text
https://huggingface.co/esatapedico/Qwen3.8-27B-NVFP4-MTP-GGUF
```

Base checkpoint:

```text
https://huggingface.co/Qwen/Qwen3.8-27B
```

#### Qwen3.8-27B DFlash2 Q4_K_M drafter

Local file:

```text
Qwen3.8-27B-DFlash2-Q4_K_M.gguf
```

GGUF metadata resolves the provenance:

```text
general.author     = "Inco AI"
general.source.url = "https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2"
```

Model lineage:

```text
Base / DFlash2 source:
https://huggingface.co/z-lab/Qwen3.8-27B-DFlash2

Q4_K_M GGUF re-quant used on this machine:
https://huggingface.co/incoai/Qwen3.8-27B-DFlash2-GGUF
```

So the running file is **Inco AI's Q4_K_M GGUF re-quant of the z-lab DFlash2 model**.

### A few flags that matter on the GX10

#### `--lazy-mode on-direct`

This matters when the combined system is operating close to the usable-memory ceiling.

It allows the runtime to avoid eagerly committing everything at startup and gives the two-slot configuration more room to coexist.

#### `--kv-unified` + q8_0 KV

Quantizing the KV cache to q8_0 materially reduces its memory footprint compared with BF16.

I have not observed a meaningful quality regression from q8_0 KV in these agentic workloads, and the memory savings are one of the reasons dual 96K contexts are practical.

#### Neural drafter + `ngram-mod`

Both slots use two speculative mechanisms:

- Flash-Next: `draft-mtp` + `ngram-mod`
- 27B: `draft-dflash` + `ngram-mod`

The neural drafter handles predictable continuation well.

`ngram-mod` adds a very cheap second path that is particularly useful when the output contains repeated structures or previously seen text.

#### `--reasoning-preserve`

This has mattered for Hermes.

Long agentic workflows often include multiple tool-call round trips. Preserving the model's reasoning state across those transitions has been much more stable in our use than rebuilding the reasoning path after every tool response.

---

## 5. Memory architecture: from "mlock and pray" to graceful pressure handling

This is the part of the design that changed most dramatically since June.

### June

The basic philosophy was:

> Lock everything. Avoid swap. Keep the whole inference stack resident.

That worked while the individual models were smaller.

It became fragile once the primary model grew to 106 GB.

Any significant host-memory spike could put the entire box into an OOM situation.

I saw that during the bring-up of the new stack.

### September

Current configuration:

```text
zswap: enabled
compressor: lzo
zpool: zsmalloc
max_pool_percent: 1

vm.swappiness=1

swapfile:
/swap.img
100G
```

The settings are persisted through `/etc/sysctl.d/99-zswap-tuning.conf` plus a small systemd unit.

The important point is **not** that swap somehow makes GPU allocations unlimited.

It does not.

The point is to give Linux somewhere to put cold, reclaimable host-side state instead of forcing an OOM event because a nearly full unified-memory system suddenly needs a little more room.

With `swappiness=1`, the kernel is reluctant to swap anonymous memory unless actual pressure develops.

zswap provides a small compressed layer before disk-backed swap.

For this workload, that has been a much better failure mode than the June "no headroom" design.

### Live snapshot — September 16

Both slots loaded and serving:

```text
Memory:
121 GiB total
~120 GiB used
~420 MiB available

Swap:
99 GiB total
21.8 GiB used

Slot A:
GPU-accounted: 84,382 MiB
RSS:           9.2 GiB
swap:          3.0 GiB

Slot B:
GPU-accounted: 22,755 MiB
RSS:           28 MiB
swap:          17.4 GiB
```

Yes, the swap numbers look alarming at first.

In practice, what matters is whether the machine is **actively thrashing**.

With a steady workload, NVMe activity is close to zero.

A slot that has been idle for a long time can incur a few seconds of extra latency when cold host-side pages fault back in. After that, the system returns to normal.

### How I distinguish useful swap from thrashing

I watch:

```bash
sar -B 1
```

and:

```bash
iostat -x 1
```

Healthy steady state:

- `pgin/s` near zero
- NVMe mostly idle
- normal decode throughput
- no sustained CPU iowait spike

Actual thrashing looks very different:

- persistent page-ins
- sustained NVMe utilization
- rising iowait
- tok/s collapsing

At that point, the working set really is too large.

The answer is not another zswap tweak.

The answer is to reduce context, reduce model size, or change the workload.

### A useful GX10 warning

On this configuration, `free -h` showing almost all 121 GiB used plus substantial swap usage is **not by itself evidence that something is wrong**.

With llama.cpp and unified memory, unused memory is not the goal.

I judge the system by:

1. response latency
2. decode throughput
3. page-in behavior
4. storage I/O
5. whether the agent workloads complete reliably

---

## 6. "Left brain / right brain" architecture

This is just my shorthand; I am not making a biological claim about the models.

The useful part is that the two slots have deliberately different jobs.

### Left brain — Flash-Next

Port `10994`

This is the primary model.

It gets:

- difficult interactive requests
- complex reasoning
- tool use
- delegation
- final response generation
- MoA aggregation

Production decode:

- median: **40.1 tok/s**
- p90: **55.9 tok/s**

It is the more capable general-purpose model in this system, despite not being the faster model in raw decode throughput.

### Right brain — Qwen3.8-27B NVFP4

Port `10993`

This is the dense/reference model.

It gets:

- cron work
- long injected context
- reference answers
- detail-sensitive work
- second-pass checking

Production decode:

- median: **44.1 tok/s**
- p90: **62.1 tok/s**

In my workloads, the dense model tends to be more literal and careful with long, instruction-heavy context, while Flash-Next is the stronger general agent.

That difference is useful.

### MoA mode

Hermes can call both.

Current preset:

```yaml
moa:
  presets:
    default:
      reference_models:
        - provider: Qwen3.8-27B-NVFP4
          model: Qwen3.8-27B-NVFP4

      aggregator:
        provider: Qwen3.8-Flash-Next
        model: Qwen3.8-Flash-Next

      reference_max_tokens: 600
```

The 27B writes a reference answer first.

Flash-Next receives that output and produces the final answer.

On difficult analytical prompts, this has been useful because the dense model often catches details that the MoE would otherwise gloss over, while Flash-Next is better at synthesis and the final response.

I have not treated this as a formal benchmark, so I would describe the quality gain as an operational observation rather than a measured universal improvement.

The `/moa` path itself is completely local.

### Cloud offload valve

I do still use cloud inference selectively.

High-volume context compression and background summarization are routed to:

```text
OpenRouter
deepseek/deepseek-v4-flash
```

That is intentional.

Earlier, I tried using very small local models for those jobs. That turned out to be a poor tradeoff; see §9.

The objective is not "zero cloud at any cost."

The objective is to keep the expensive local slots available for the work where they provide the most value.

---

## 7. Real production tok/s

I wanted better numbers than the spot measurements in my June post.

These are taken from actual llama.cpp server logs under the system's normal workload.

### Source

llama.cpp `print_timing` lines:

- `tg` — decode speed
- `tg_3s` — short trailing decode-rate measurement used here for the distributions

### Measurement window

**2026-09-14 through 2026-09-16 CST**

The current servers started around 02:18 / 02:27 CST on September 16, and I included the tail of the immediately preceding sessions to increase sample size.

This is normal production traffic:

- Hermes gateway use
- cron jobs
- both models resident
- varying context depth
- normal speculative decoding

It is **not** an isolated single-stream benchmark.

### Sample counts

Flash-Next:

```text
current session: 45,349 tg_3s samples
prior tail:       2,206
```

27B NVFP4:

```text
current session: 84,950 tg_3s samples
prior tail:       4,413
```

Total: approximately **137,000 measurements**.

### Results

| Slot | p10 | median | mean | p90 | max |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Flash-Next @96K | 21.8 | **40.1** | 39.6 | **55.9** | 103.5 |
| Qwen3.8-27B NVFP4 @96K | 21.8 | **44.1** | 43.0 | **62.1** | 145.3 |

### Draft acceptance

Same window, using the server's `draft acceptance` log lines.

Flash-Next:

```text
n = 49,801
median = 0.79
mean   = 0.79
p90    = 0.99
```

27B:

```text
n = 49,099
median = 0.82
mean   = 0.80
p90    = 0.98
```

### Why the distribution is so wide

This is what real agent traffic looks like.

Decode rate changes with:

- context depth
- simultaneous prefill activity
- task type
- speculative-decoding acceptance
- whether output is novel prose, code, or repetitive structured text

The ~22 tok/s p10 floor represents the kind of performance I see with deep context and competing load.

The 100+ tok/s maxima are short-context speculative-decoding bursts.

For capacity planning, I care much more about the median and lower end of the distribution than the maximum screenshot number.

One other caveat:

The June "~60 / ~25 tok/s" numbers were spot measurements.

These September numbers are distributions over roughly 137,000 production samples.

They are useful for showing how the system behaves today, but they are not a controlled apples-to-apples benchmark against June.

---

## 8. The long-conversation drift problem — and what actually fixed it

One of the stranger problems during this phase did not show up immediately after startup.

The model could work normally through a sequence of Hermes interactions and then, after a long conversation involving repeated tool calls and speculative decoding, begin to lose coherence.

The symptom was difficult to diagnose because:

- the server remained alive;
- generation speed could remain normal;
- short fresh conversations could still look fine;
- the degradation appeared primarily after accumulated agent/tool state.

At first, this looked like a model-quality or attention problem.

Our debugging eventually traced it instead to **state handling around speculative decoding and the recurrent portions of the hybrid model**.

Three changes ended up mattering.

### 1. Hybrid recurrent checkpoint restore

Commit:

```text
fa2970869
```

The largest issue was that recurrent-state snapshots were not being restored correctly through the checkpoint/rollback path used around speculative decoding.

For a conventional transformer, KV state dominates the problem.

For a hybrid model with recurrent/SSM state, restoring the visible KV state is not enough if the recurrent state no longer corresponds to it.

The practical symptom was exactly what we were seeing: a conversation could remain superficially functional while continuations gradually stopped making sense.

This fix went in on **September 10**.

### 2. MTP draft-head attachment

Commit:

```text
7e5fd8901
```

On September 15, I switched the MTP draft borrowing path to Unsloth's tensor-presence gate.

This cleaned up how the shared MTP head attaches to the split Flash-Next model and avoided the partially initialized state we had encountered with the earlier path.

### 3. Preserve reasoning state across tool calls

Current server flag:

```text
--reasoning-preserve
```

Hermes workflows can contain many model → tool → model transitions.

Preserving reasoning state across those transitions has been noticeably more stable than repeatedly reconstructing the reasoning path after each tool response.

### Current result

Since the recurrent checkpoint fix went in on September 10, I have not observed a recurrence of the long-conversation failure mode in the current production sessions.

The current September 16 server session also shows healthy speculative decoding on Flash-Next:

```text
draft-acceptance samples: ~49,800
median acceptance:        0.79
mean acceptance:          0.79
p90 acceptance:           0.99
```

I would **not** claim that the acceptance-rate numbers prove the drift problem is solved — they measure a different thing.

What matters more is that the same Hermes workload that previously exposed the failure has now been running without the observed coherence degradation since the checkpoint-state fix.

This was probably the most misleading bug in the Phase 2 build because it initially looked like "the model gets worse over time."

In our case, the model was not changing.

**The state being handed back to it was wrong.**

---

## 9. What did not work for us

These were some of the more useful lessons from the last three months.

### 1. Tiny local utility models were a false economy

I ran Qwen3.5-4B and Qwen3.5-2B for weeks as lightweight utility/compression models.

I eventually removed both.

Together, they consumed roughly 6 GB that the primary model needed more.

More importantly, the 2B model was not reliable enough for context compression.

A lossy summary is worse than no summary when a stronger model later treats that compressed state as ground truth.

We saw exactly that: errors introduced during compression propagated into the expensive model's later reasoning.

I now offload those high-volume, low-priority summarization jobs instead.

### 2. mlock + effectively zero swap headroom stopped working once the flagship model became huge

That was my June philosophy.

It was fine for the old setup.

With this one, it turned normal memory spikes into possible OOM events.

Allowing controlled reclamation has been significantly more robust for this workload.

### 3. "99% RAM used" is not a useful health metric by itself

On a unified-memory inference machine, high utilization is expected.

I care about:

```text
pgin/s
iowait
NVMe utilization
latency
tok/s
```

Those tell me whether memory pressure is actually hurting the workload.

### 4. I stopped chasing 128K × 2

June made me think in terms of maximum context.

September made me think in terms of **usable context**.

I run:

```text
96K + 96K
```

deliberately.

That gives me two stable long-context models rather than one configuration that looks better on paper but has much less operational headroom.

For my workload, that is a better trade.

### 5. vLLM worked, but llama.cpp fit this workload better

I did get Qwen3.8-Flash-Next serving under vLLM with the PLE setup at roughly **32 tok/s**.

So this is not a claim that vLLM "doesn't work" on the GX10.

The issue is workload shape.

Our agent system:

- can idle for long periods
- wakes in bursts
- uses very long context
- benefits heavily from MTP / DFlash2 / ngram speculative decoding
- runs very close to the unified-memory limit

For that specific workload, the current llama.cpp setup has been a better fit.

### 6. Agent-state preservation matters as much as raw inference speed

`--reasoning-preserve` has been important for longer Hermes tool loops.

This was one of the broader lessons of the project:

A model can have excellent tok/s and still be a poor production agent if state across repeated tool calls is unreliable.

---

## 10. What the box actually does now

This is no longer primarily a benchmarking machine.

Hermes Agent is running continuously.

Current state:

```text
Hermes Agent: v0.21.0
Python:       3.11.16
gateway:      6d+ continuous uptime at snapshot

cron jobs:
74 defined
26 currently enabled
```

The enabled jobs cover combinations of:

- research
- publishing
- monitoring
- outreach
- recurring internal workflows
- context maintenance
- tool-driven agent tasks

The two local models handle the reasoning-heavy portions, with selective OpenRouter offload for high-volume compression/summarization.

One lesson from the last several months has been to use browser automation **only when it is actually necessary**.

If an API or CLI can perform the same operation deterministically, I prefer that.

Browser control is still useful, but every UI transition introduces another probabilistic failure point.

The more production-oriented this system became, the more we moved from:

> "Can the AI operate the browser?"

 toward:

> "What is the most reliable interface we can give the AI?"

That change has mattered as much as some model upgrades.

---

## 11. Supporting stack

Verified at the September 16 snapshot:

```text
Hermes Agent      v0.21.0
Python            3.11.16
openai            2.24.0
fastapi           0.133.1
httpx             0.28.1
uvicorn           0.41.0
pydantic          2.13.4
Node              v22
comfy-cli          1.10.3
```

GUI/browser support:

- camofox anti-detection browser
- noVNC on `:6080`
- RDP on `:3389`

I do not currently use Docker on this box.

That is not because Docker cannot work here. I simply found that running the services directly made unified-memory accounting and debugging easier, while containerization provided little benefit for this particular single-purpose machine.

ComfyUI remains separate from the agent architecture. It is installed on the external NVMe and is currently offline while I check that drive's health, so none of the performance figures in this post depend on it.

---

## 12. Reproduction / troubleshooting checklist

If you are trying to reproduce this on a GX10 / DGX Spark:

1. Use BIOS `GX10DGX.0104` or later.
2. Install the CUDA 13.3 toolchain and the 610.xx open driver.
3. Clone the documented Unsloth and upstream llama.cpp remotes.
4. Check out exact base `66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3`.
5. Download `nv-forum-patches.zip` from this post.
6. Verify the five patch files against `PATCHES-SHA256.txt`.
7. Apply patch 01.
8. Merge `ccc827348` for the Unsloth MTP/on-direct lineage.
9. Apply patch 02.
10. Merge upstream `1bc7a5af0`.
11. Apply patches 03–05 in order.
12. Confirm the resulting HEAD is `be9d75bd932947c20f36acdb7668c9b330253ca7`.
13. Build with the GB10-specific CMake configuration in §3.
14. Configure the memory-pressure strategy in §5.
15. Download the exact model/drafter combinations and sources listed in §4.
16. Start with q8_0 KV and 96K contexts before attempting larger contexts.
17. Bring up one model first and validate it.
18. Start the second slot and monitor actual paging/I/O behavior rather than judging the system from `free -h` alone.
19. Verify speculative decoding is operating by checking draft acceptance.
20. Evaluate sustained performance from server logs, not a single short generation.

Current GPU-accounted memory is roughly:

```text
Slot A: ~84 GB
Slot B: ~23 GB
```

If decode throughput is dramatically below the ranges in §7, my troubleshooting order is:

```text
1. Check page-in / storage pressure.
2. Check context depth.
3. Check draft acceptance.
4. Confirm the intended drafter actually loaded.
5. Confirm the CUDA architecture/build.
6. Only then start changing sampling or model settings.
```

---

## 13. Where this leaves the project

The interesting part for me is no longer peak tok/s.

Three months ago, the achievement was simply getting a multi-model Hermes system stable on the GX10.

Today, the same machine is running a 106 GB primary model and a 27B dense model side by side, both with 96K context, both available to the agent harness, and both producing usable throughput under real scheduled workloads.

More importantly, the problems have changed.

In Phase 1, most failures were obvious:

- the machine crashed;
- the model would not load;
- CUDA broke;
- performance collapsed;
- a workflow simply failed.

Phase 2 exposed a harder class of problems:

- recurrent state that restores incorrectly;
- speculative decoding interacting with long-running agent state;
- small utility models corrupting context through bad compression;
- apparently alarming swap usage that is actually harmless;
- browser automation that works but is less reliable than an API;
- systems that can appear healthy while their internal state is drifting away from what the model expects.

That is probably the biggest difference between the two phases.

**Phase 1 was about making local agents possible.**

**Phase 2 has been about making them operational.**

The GX10 no longer feels like an experiment I am constantly trying to keep alive.

It feels like infrastructure.

And, in some ways, that has made the remaining problems more interesting rather than less.

---

*All performance and memory numbers above are from a live production system on September 16, 2026. Workload mix, context depth, and speculative-decoding acceptance can move the numbers substantially, so I would expect other systems to vary.*

*Happy to answer questions about the QSA path, hybrid/recurrent fixes, MTP/DFlash2 behavior, or the memory configuration. Those are where most of the interesting problems were.*
