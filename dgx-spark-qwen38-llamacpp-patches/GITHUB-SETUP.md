# Create the GitHub repository

After unzipping this package into an empty directory:

```bash
git init
git add .
git commit -m "Initial DGX Spark Qwen3.8 llama.cpp patch release"
git branch -M main
```

Create an empty public repository on GitHub, then connect and push it:

```bash
git remote add origin https://github.com/YOUR-ACCOUNT/YOUR-REPO.git
git push -u origin main
```

Suggested repository name:

```text
dgx-spark-qwen38-llamacpp-patches
```

Suggested description:

```text
Reproducible llama.cpp patch set and source lineage for a production ASUS GX10 / NVIDIA DGX Spark running Qwen3.8 Flash-Next + Qwen3.8-27B.
```

Suggested first release tag:

```text
phase2-2026-09-16
```

After pushing, create a GitHub Release for that tag if you want a frozen downloadable snapshot in addition to the live repository.
