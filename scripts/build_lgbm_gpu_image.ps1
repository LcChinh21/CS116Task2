$ErrorActionPreference = "Stop"

$RepoRoot = [string](Resolve-Path (Join-Path $PSScriptRoot ".."))
$ImageName = if ($env:CS116_LGBM_GPU_IMAGE) { $env:CS116_LGBM_GPU_IMAGE } else { "cs116task2-lgbm-gpu:cuda11.7" }
$Dockerfile = Join-Path $RepoRoot "docker\Dockerfile.lgbm-gpu"

docker build `
  -f $Dockerfile `
  -t $ImageName `
  $RepoRoot
