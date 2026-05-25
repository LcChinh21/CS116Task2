# Submit Plan

1. `outputs/submission_best_38_8.pkl`
   - validation MAPE: 51.548672 existing Dec validation from best 38.8 run
   - scale/postprocess: control artifact, no overwrite
   - ly do nop: giu best public score 38.8 lam control
2. `outputs/submission_raw_only_scale_best.pkl`
   - validation MAPE: 59.067970
   - scale/postprocess: global scale 0.700
   - ly do nop: unweighted raw LightGBM; local best global scale
3. `outputs/submission_weighted_inv_y_scale_best.pkl`
   - validation MAPE: 51.345707
   - scale/postprocess: global scale 1.000
   - ly do nop: MAPE-like inv_y sample weights; local best global scale
4. `outputs/submission_weighted_inv_sqrt_y_scale_best.pkl`
   - validation MAPE: 55.463587
   - scale/postprocess: global scale 0.900
   - ly do nop: milder inv_sqrt_y sample weights; local best global scale
5. `outputs/submission_group_scale_best.pkl`
   - validation MAPE: 51.344541
   - scale/postprocess: group scales {"high_sale": 0.975, "low_sale": 1.0, "mid_sale": 1.0}
   - ly do nop: best local candidate among group-scale improvements and raw-baseline blends
