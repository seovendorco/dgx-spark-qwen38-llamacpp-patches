#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PATCH_DIR="$REPO_ROOT/patches"
TARGET=${1:-"$REPO_ROOT/../llama.cpp-gx10"}

BASE_COMMIT=66bb57b3f905f5dc6b8125fac1d737d52ab3a8e3
UNSLOTH_MTP_COMMIT=ccc827348
UPSTREAM_COMMIT=1bc7a5af0
RECORDED_PRODUCTION_HEAD=be9d75bd932947c20f36acdb7668c9b330253ca7

if [ -e "$TARGET" ]; then
    echo "Error: target already exists: $TARGET" >&2
    echo "Choose a new empty path or remove it yourself." >&2
    exit 1
fi

cd "$PATCH_DIR"
sha256sum -c PATCHES-SHA256.txt
cd - >/dev/null

git clone https://github.com/unslothai/llama.cpp.git "$TARGET"
cd "$TARGET"

git remote add upstream https://github.com/ggml-org/llama.cpp.git
git fetch origin
git fetch upstream

git checkout "$BASE_COMMIT"

git am "$PATCH_DIR/01-server-fix-hybrid-recurrent-checkpoint-restore.patch"

git merge --no-edit "$UNSLOTH_MTP_COMMIT"

git am "$PATCH_DIR/02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch"

git merge --no-edit "$UPSTREAM_COMMIT"

git am "$PATCH_DIR/03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch"
git am "$PATCH_DIR/04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch"
git am "$PATCH_DIR/05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch"

printf '\nReconstruction complete.\n'
printf 'Current HEAD: %s\n' "$(git rev-parse HEAD)"
printf 'Recorded production integration HEAD: %s\n' "$RECORDED_PRODUCTION_HEAD"
printf '\nNote: commit IDs can differ when merge/committer metadata differ even when the resulting source tree is equivalent.\n'
printf 'See README.md and docs/BUILD.md in the patch repository for verification guidance.\n'
