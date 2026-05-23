# CS116Task2

Sale Forecasting Task 2

## Cloud GPU Runbook

Tested base image:

```text
pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

Hardware used:

```text
GPU: NVIDIA RTX A5000, 16-24GB VRAM
CPU: 8 cores
RAM: 30GB+
```

## One-time environment setup

Install basic tools:

```bash
apt-get update
apt-get install -y curl bzip2 ca-certificates git build-essential cmake \
  libboost-dev libboost-system-dev libboost-filesystem-dev libboost-chrono-dev \
  ocl-icd-libopencl1 ocl-icd-opencl-dev clinfo libgomp1
```

Install micromamba:

```bash
cd ~
rm -rf bin
curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=/opt/micromamba
./bin/micromamba shell init -s bash -r /opt/micromamba
source ~/.bashrc
```

Create RAPIDS/cuDF environment:

```bash
micromamba create -y -n rapids-feature \
  -c rapidsai -c conda-forge -c nvidia \
  python=3.10 cudf=25.06 cuda-version=11.8
micromamba activate rapids-feature
```

Install project dependencies:

```bash
cd ~/CS116Task2
pip install -r requirements.txt
```

## Feature engineering on GPU

Run with cuDF required, so the script stops if it would fall back to CPU:

```bash
micromamba activate rapids-feature
cd ~/CS116Task2
FEATURES_REQUIRE_GPU=1 python src/features.py
```

Expected outputs:

```text
outputs/features_val_nov.parquet
outputs/features_val_dec.parquet
outputs/features_predict_jan2026.parquet
outputs/features_train.parquet
```

## LightGBM GPU setup

LightGBM `device_type=gpu` uses OpenCL. Enable the NVIDIA OpenCL ICD:

```bash
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ldconfig
clinfo | grep -E "Number of platforms|Platform Name|Device Name" -A2
```

Expected:

```text
Number of platforms 1
Platform Name NVIDIA CUDA
Device Name NVIDIA RTX A5000
```

Build and install LightGBM with GPU support:

```bash
micromamba activate rapids-feature
cd /tmp
rm -rf LightGBM
git clone --recursive --branch v4.6.0 --depth 1 https://github.com/microsoft/LightGBM.git
cd LightGBM
cmake -B build -S . -DUSE_GPU=ON \
  -DCMAKE_C_COMPILER=/usr/bin/gcc \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++
cmake --build build -j8
sh ./build-python.sh install --precompile
```

Quick check:

```bash
python - <<'PY'
import lightgbm
print(lightgbm.__file__)
print(hasattr(lightgbm, "train"), hasattr(lightgbm, "Dataset"))
PY
```

## Train LightGBM on GPU

```bash
micromamba activate rapids-feature
cd ~/CS116Task2

LGBM_USE_GPU=1 \
LGBM_DEVICE_TYPE=gpu \
LGBM_MAX_BIN=63 \
LGBM_GPU_PLATFORM_ID=0 \
LGBM_GPU_DEVICE_ID=0 \
OMP_NUM_THREADS=8 \
python src/train_lgbm.py
```

Expected outputs:

```text
models/lgbm_poisson.txt
models/lgbm_tweedie.txt
models/lgbm_l1.txt
outputs/predictions_lgbm.parquet
outputs/validation_predictions.csv
reports/model_results.md
```

Monitor GPU:

```bash
watch -n 2 nvidia-smi
```

## Blend and final submission

```bash
micromamba activate rapids-feature
cd ~/CS116Task2

python src/blend.py
python src/postprocess.py
python src/check_submission.py
```

Final submission file:

```text
outputs/submission_final.pkl
```

Final submission schema:

```text
location, item_id, quantity_pred
```

## Common fixes

If `features.py` says cuDF is missing:

```bash
micromamba activate rapids-feature
python -c "import cudf; print(cudf.__version__)"
```

If LightGBM says `GPU Tree Learner was not enabled`, rebuild LightGBM with:

```bash
cd /tmp/LightGBM
cmake -B build -S . -DUSE_GPU=ON
cmake --build build -j8
sh ./build-python.sh install --precompile
```

If LightGBM says `No OpenCL device found`, recreate the ICD file:

```bash
mkdir -p /etc/OpenCL/vendors
echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd
ldconfig
clinfo | grep -E "Number of platforms|Platform Name|Device Name" -A2
```

If GPU memory is too tight on 16GB, reduce `config.yaml`:

```yaml
LGBM_COMMON:
  num_leaves: 63
  n_estimators: 800

LGBM_GPU:
  max_bin: 31
  gpu_use_dp: false
```

## Docker GPU training setup

Optional local Docker path. Base image:

```text
pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime
```

Build the GPU image:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_lgbm_gpu_image.ps1
```

Run LightGBM training on GPU later:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\train_lgbm_gpu.ps1
```

The runner sets `LGBM_USE_GPU=1` and mounts this repo to `/workspace`.
