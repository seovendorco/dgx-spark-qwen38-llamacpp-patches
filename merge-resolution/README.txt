Pre-resolved files for step 4 of the reconstruction (git merge 1bc7a5af0).

After the merge reports conflicts:

  cp /path/to/merge-resolution/qwen4exp.cpp     src/models/qwen4exp.cpp
  cp /path/to/merge-resolution/llama-bench.cpp  tools/llama-bench/llama-bench.cpp
  git add -A
  git commit
  # then continue with patches 02-05

These blobs are taken from the production tree (commit 7e5fd8901, immediately
before patches 02-05) and were used in the verified 2026-09-23 reconstruction
that produced tree 5a421ae1c9948aca981aa5f7db59e5d2ec0ceeae.
