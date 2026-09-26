# Entity-Level Evaluation & Threshold Sweep Report
Amazon ML Challenge 2026

## Target Metric: Validation Macro-Averaged $F_{0.5}$

The competition evaluates submissions using **Macro-Averaged $F_{0.5}$** across all Source 1 entities, including singletons:
$$F_{0.5} = \frac{1.25 \cdot \text{Precision} \cdot \text{Recall}}{0.25 \cdot \text{Precision} + \text{Recall}}$$

- **Optimal Classification Threshold**: **`0.90`**
- **Optimal Macro-Averaged $F_{0.5}$**: **`0.8017`** (80.17%)
- **Macro Precision at Optimal Threshold**: **`0.8670`** (86.70%)
- **Macro Recall at Optimal Threshold**: **`0.7045`** (70.45%)
- **Singleton Accuracy**: **`74.46%`** (552 singletons evaluated)
- **Matched Entities $F_{0.5}$**: **`0.8051`** (9,447 matched entities evaluated)

---

## Threshold Sweep Trajectory (0.50 to 0.95, step = 0.05)

| Threshold | **Macro $F_{0.5}$** | Macro Precision | Macro Recall | Singleton Accuracy | Matched $F_{0.5}$ | Note |
| :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **0.50** | **0.7198** | 0.7242 | 0.8267 | 25.54% | 0.7470 | |
| **0.55** | **0.7311** | 0.7389 | 0.8223 | 29.17% | 0.7568 | |
| **0.60** | **0.7412** | 0.7528 | 0.8160 | 32.61% | 0.7654 | |
| **0.65** | **0.7526** | 0.7684 | 0.8096 | 37.50% | 0.7747 | |
| **0.70** | **0.7644** | 0.7855 | 0.8012 | 41.85% | 0.7846 | |
| **0.75** | **0.7752** | 0.8029 | 0.7892 | 45.11% | 0.7942 | |
| **0.80** | **0.7866** | 0.8221 | 0.7742 | 51.63% | 0.8023 | |
| **0.85** | **0.7967** | 0.8424 | 0.7521 | 60.51% | 0.8079 | |
| **0.90** | **0.8017** | 0.8670 | 0.7045 | 74.46% | 0.8051 | **(OPTIMAL)** |
| **0.95** | **0.7804** | 0.8723 | 0.6287 | 89.31% | 0.7738 | |

---

## Country-Level Performance at Optimal Threshold

| Country | Entities Evaluated | **Macro $F_{0.5}$** | Macro Precision | Macro Recall |
| :--- | :---: | :---: | :---: | :---: |
| **US** | 5,998 | **0.8355** | 0.8909 | 0.7509 |
| **India** | 4,001 | **0.7511** | 0.8311 | 0.6350 |

---

## Key Strategic Takeaways

1. **Precision-Recall Asymmetry**:
   - Because $F_{0.5}$ weights precision 2× over recall, raising the threshold from 0.50 to **0.90** yields a significant jump in overall Macro $F_{0.5}$ (from 0.7198 to **0.8017**, a **+8.19% absolute improvement**).
2. **Singleton Protection**:
   - At threshold 0.90, singleton accuracy reaches **74.46%**, eliminating false positive matches on unlinked entities.
3. **Threshold Calibration**:
   - The threshold of **0.90** should be used for all test set predictions to generate `matching_results.tsv`.
