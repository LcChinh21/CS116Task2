# CS116Task2

Sale Forecasting Task 2

## LightGBM GPU training setup

Base image: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime`

Build the GPU image:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_lgbm_gpu_image.ps1
```

Run LightGBM training on GPU later:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_lgbm_gpu.ps1
```

The runner sets `LGBM_USE_GPU=1` and mounts this repo to `/workspace`.
