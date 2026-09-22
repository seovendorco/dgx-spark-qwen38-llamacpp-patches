#!/usr/bin/env sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/.." && pwd)
PATCH_DIR="$REPO_ROOT/patches"
TARGET=${1:-"$REPO_ROOT/../llama.cpp-gx10"}

PUBLIC_BASE=a9e9c3c5f
UPSTREAM_COMMIT=1bc7a5af0
EXPECTED_TREE=5a421ae1c9948aca981aa5f7db59e5d2ec0ceeae
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

git checkout "$PUBLIC_BASE"

# Recreate the private base commits (lazy/direct-read machinery).
git am "$PATCH_DIR/00a-base-qwen4exp-direct-reads-for-lazy-PLE-table.patch"
git am "$PATCH_DIR/00b-base-lazy-share-direct-read-machinery-gemma4.patch"

git am "$PATCH_DIR/01-server-fix-hybrid-recurrent-checkpoint-restore.patch"

# Merge the recorded upstream tree. Two files are expected to conflict
# (src/models/qwen4exp.cpp, tools/llama-bench/llama-bench.cpp): the private
# lazy/direct-read work vs upstream changes. This script does not guess a
# resolution; resolve manually keeping BOTH the lazy/direct-read additions
# and the upstream changes, commit, then re-run the remaining `git am` steps.
if ! git merge --no-edit "$UPSTREAM_COMMIT"; then
    echo "" >&2
    echo "Merge conflict (expected for qwen4exp.cpp / llama-bench.cpp)." >&2
    echo "Resolve both files keeping the lazy-mode (on-direct) additions AND" >&2
    echo "the upstream changes, then: git add -A && git commit" >&2
    echo "and apply patches 02-05 with git am." >&2
    exit 2
fi

git am "$PATCH_DIR/02-llama-switch-MTP-draft-borrow-to-unsloth-s-tensor-pr.patch"
git am "$PATCH_DIR/03-qwen4exp-fix-MTP-hc_head_norm-tensor-layout-after-up.patch"
git am "$PATCH_DIR/04-ggml-qwen4exp-fused-gated-hc_pre-and-identity-comb-h.patch"
git am "$PATCH_DIR/05-cuda-fattn-per-query-tile-sparse-index-lists-enable-.patch"

ACTUAL_TREE=$(git rev-parse HEAD^{tree})

printf '\nReconstruction complete.\n'
printf 'Current HEAD:  %s\n' "$(git rev-parse HEAD)"
printf 'Source tree:   %s\n' "$ACTUAL_TREE"
printf 'Expected tree: %s\n' "$EXPECTED_TREE"
printf 'Recorded production integration HEAD: %s\n' "$RECORDED_PRODUCTION_HEAD"

if [ "$ACTUAL_TREE" = "$EXPECTED_TREE" ]; then
    printf '\nOK: source tree is byte-identical to the production tree.\n'
else
    printf '\nWARNING: tree differs from the recorded production tree.\n' >&2
    printf 'If you resolved the merge manually, compare file contents before building.\n' >&2
    exit 3
fi
