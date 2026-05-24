# 20GB RAM / 4GB VRAM Pipeline

Pipeline trong thư mục này chạy theo checkpoint để tránh làm lại từ đầu sau khi lỗi hoặc bị kill.

## Chạy full profile nhẹ

```bash
micromamba activate rapids-feature
cd /root/CS116Task2
bash srcoptimized/run_20gb.sh --stage all
```

Default dành cho máy khoảng 12 core, 20GB RAM, GPU 4GB:

```text
OPT_MAX_TRAIN_ROWS=700000
OPT_MAX_FINAL_TRAIN_ROWS=1000000
OPT_MAX_EVAL_ROWS=350000
LGBM_MAX_BIN=31
OPT_LGBM_TREES=500
OPT_LGBM_LEAVES=63
OPT_RUN_CATBOOST=0
```

CatBoost tắt mặc định vì 4GB VRAM thường không đủ ổn định.

## Chạy từng phần

```bash
bash srcoptimized/run_20gb.sh --stage aggregate
bash srcoptimized/run_20gb.sh --stage features
bash srcoptimized/run_20gb.sh --stage train
bash srcoptimized/run_20gb.sh --stage final
bash srcoptimized/run_20gb.sh --stage predict
```

Checkpoint nằm ở:

```text
outputs/srcoptimized_cache/
```

Xem checkpoint nào đã có:

```bash
bash srcoptimized/run_20gb.sh --stage status
```

Chạy lại một stage từ đầu:

```bash
bash srcoptimized/run_20gb.sh --stage train --force
```

## Nếu vẫn bị kill

Giảm tiếp sample:

```bash
OPT_MAX_TRAIN_ROWS=400000 \
OPT_MAX_FINAL_TRAIN_ROWS=600000 \
OPT_MAX_EVAL_ROWS=200000 \
OPT_LGBM_TREES=350 \
bash srcoptimized/run_20gb.sh --stage train --force
```

Sau đó chạy tiếp:

```bash
bash srcoptimized/run_20gb.sh --stage final --force
bash srcoptimized/run_20gb.sh --stage predict
```

## Output

File cuối cùng vẫn là:

```text
outputs/submission_final.csv
```
