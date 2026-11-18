| Model | Algorithm family | Key configuration | CV macro F1 | Top-1 accuracy (student test) | Route continuity (classifier only, student test) | Student-test macro F1 | Log loss (student test) | Inference time (s, 201,662 candidate rows) |
|---|---|---|---|---|---|---|---|---|
| LightGBM | Boosting | {"n_estimators": 600, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "subsample": 0.8, "subsample_freq": 1, "colsample_bytree": 0.8, "reg_lambda": 1.0} | 0.9396 ± 0.0141 | 0.9033 | 0.9756 | 0.9428 | 0.0472 | 0.760 |
| LightGBM (tuned) | Boosting | {"n_estimators": 875, "learning_rate": 0.023619345141514477, "num_leaves": 60, "min_child_samples": 24, "subsample": 0.8565405722143823, "subsample_freq": 1, "colsample_bytree": 0.4033663838842876, "reg_lambda": 0.4782843849439511, "reg_alpha": 0.013176449521576582} | 0.9401 ± 0.0145 | 0.9004 | 0.9744 | 0.9427 | 0.0474 | 1.344 |
| Random forest | Ensemble (bagging) | {"n_estimators": 200, "min_samples_leaf": 20, "max_features": "sqrt"} | 0.9390 ± 0.0138 | 0.8972 | 0.9691 | 0.9414 | 0.0491 | 0.421 |
| MLP | Neural network | {"hidden_layer_sizes": [64, 32], "alpha": 0.0001, "batch_size": 1024, "learning_rate_init": 0.001, "max_epochs": 40, "patience": 5} | 0.9353 ± 0.0137 | 0.8908 | 0.9712 | 0.9357 | 0.0541 | 0.062 |
| Logistic regression | Linear | {"C": 1.0, "max_iter": 2000} | 0.9250 ± 0.0142 | 0.8899 | 0.9731 | 0.9232 | 0.0697 | 0.022 |
| KNN | Instance-based | {"n_neighbors": 50, "weights": "distance"} | 0.9140 ± 0.0151 | 0.8698 | 0.9669 | 0.9166 | 0.0759 | 43.167 |
| Gaussian NB | Probabilistic | {"var_smoothing": 1e-09} | 0.8516 ± 0.0060 | 0.8611 | 0.9601 | 0.8370 | 0.5814 | 0.070 |
| Baseline: distance + heading | Non-learned rule | rule-based (no training) | 0.9161 ± 0.0160 | 0.8539 | see results/hmm | 0.9192 | N/A (hard rule, no probabilities) | 0.047 |
| Baseline: nearest arc | Non-learned rule | rule-based (no training) | 0.7561 ± 0.0317 | 0.5352 | see results/hmm | 0.7432 | N/A (hard rule, no probabilities) | 0.001 |
