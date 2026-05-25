# Submit Plan

1. `outputs/submission_best_38_9.csv`
   - validation MAPE: 51.344541
   - public score: 38.9
   - scale/postprocess: control from previous best `submission_group_scale_best.csv`; 3-group scales `{"high_sale": 0.975, "low_sale": 1.0, "mid_sale": 1.0}`
   - model config: inv_y, raw_only, previous weighted candidate run
   - reason: keep current leaderboard best as control; do not overwrite

2. `outputs/submission_larger_1800k_gpu_inv_y_raw_scale_best.csv`
   - validation MAPE: 51.180930
   - scale/postprocess: global scale 1.000
   - model config: GPU LightGBM, inv_y, raw_only, train rows 1.8M, final rows 2.6M, eval rows 700k, trees 900, leaves 63, max_bin 31, CatBoost off
   - reason: strongest pure raw inv_y local score; low-risk because no group extrapolation

3. `outputs/submission_current_inv_y_5group_scale.csv`
   - validation MAPE: 51.504034
   - scale/postprocess: 5-group scales `{"very_low": 1.0, "low": 1.0, "mid": 1.0, "high": 0.925, "very_high": 1.1}`
   - model config: inv_y, raw_only, current cache, 137 features, CatBoost off
   - reason: refined 5-group scale beats current global and legacy 3-group locally

4. `outputs/submission_larger_1800k_gpu_inv_y_5group_scale.csv`
   - validation MAPE: 51.150370
   - scale/postprocess: 5-group scales `{"very_low": 1.0, "low": 1.0, "mid": 1.0, "high": 0.95, "very_high": 1.125}`
   - model config: GPU LightGBM, inv_y, raw_only, train rows 1.8M, final rows 2.6M, eval rows 700k, trees 900, leaves 63, max_bin 31, CatBoost off
   - reason: best local candidate; 5-group scale beats global 51.180930 and legacy 3-group 51.178582

5. `outputs/submission_larger_1800k_gpu_inv_y_scale_neighbor.csv`
   - validation MAPE: 51.587211
   - scale/postprocess: global scale neighbor 1.025
   - model config: same GPU larger-sample inv_y raw_only model as submit 2/4
   - reason: fallback slot because event-feature variant is not validated yet; include only after higher-priority files if using all 5 submissions
