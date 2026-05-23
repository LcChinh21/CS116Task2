$ErrorActionPreference = "Stop"

$RepoRoot = [string](Resolve-Path (Join-Path $PSScriptRoot ".."))
$ImageName = if ($env:CS116_LGBM_GPU_IMAGE) { $env:CS116_LGBM_GPU_IMAGE } else { "cs116task2-lgbm-gpu:cuda11.7" }

docker run --rm `
  --gpus all `
  --shm-size=8g `
  -e NVIDIA_VISIBLE_DEVICES=all `
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility `
  -e LGBM_USE_GPU=1 `
  -e LGBM_DEVICE_TYPE=gpu `
  -e LGBM_MAX_BIN=63 `
  -e LGBM_GPU_PLATFORM_ID=0 `
  -e LGBM_GPU_DEVICE_ID=0 `
  -v "${RepoRoot}:/workspace" `
  -w /workspace `
  $ImageName `
  python src/train_lgbm.py
