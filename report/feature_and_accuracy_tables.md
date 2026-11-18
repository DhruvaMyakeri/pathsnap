# PathSnap: features and training vs test accuracy

## 2.2.3 Features and why they were chosen

Every (fix, candidate arc) row gets 23 features in five groups.

| Group | Features | Why |
| --- | --- | --- |
| Distance | distance to the arc, rank, distance minus the nearest candidate's, ratio to it, number of candidates, number within 10 m of the nearest | proximity matters most relative to the other candidates of the same fix |
| Direction | heading difference over ±2 s and ±5 s windows, cosine of the ±2 s difference | the only signal that separates twin arcs; set to a neutral 90° when the vehicle moved less than 1 m |
| Motion | speed, displacement, path straightness over both windows, time gap to the previous and next fix | tells the model when heading is trustworthy (stops, stationary jitter) |
| Arc and network | arc length, projection position along the arc, distance beyond the arc's end, two-way flag, out-degree of the end node, in-degree of the start node | short arcs and busy junctions behave differently |
| Context | mean and maximum distance of the 2 fixes on each side to the same arc | a single noisy fix may sit near a side street; its neighbours rarely do |

## 3.2 Training vs test accuracy (Top-1)

- **In-sample:** the final model scored on the training rows it was fitted on.
- **Cross-validated:** each training trip scored by a model that never saw it.
- **Student test:** the 10 unseen trips.

Source: `results/models/train_vs_test.csv`.

| Model | Training, in-sample | Training, cross-validated | Student test | In-sample minus test |
| --- | --- | --- | --- | --- |
| LightGBM | 0.960 | 0.899 | 0.903 | +0.057 |
| Random forest | 0.929 | 0.894 | 0.897 | +0.032 |
| MLP | 0.909 | 0.894 | 0.891 | +0.018 |
| Logistic regression | 0.894 | 0.890 | 0.890 | +0.004 |
| KNN | 1.000 | 0.870 | 0.870 | +0.130 |
| Gaussian Naive Bayes | 0.872 | 0.866 | 0.861 | +0.011 |
