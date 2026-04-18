# VERS Data Strategy & Evaluation Framework

## 1. Dataset Requirements

### Multi-Dimensional Capture Protocol

For production-grade gesture and emotion recognition, training data must span:

| Dimension | Minimum Variance | Examples |
|---|---|---|
| **Users** | ≥ 30 distinct subjects | Age 18–65, diverse skin tones |
| **Lighting** | ≥ 5 conditions | Daylight, fluorescent, dim, backlit, outdoor shade |
| **Angles** | ≥ 3 camera positions | Frontal (0°), ±30° lateral, ±15° vertical |
| **Backgrounds** | ≥ 4 environments | Office, outdoor, hospital corridor, crowd |
| **Hand sizes** | Natural variance | Small, medium, large hands |

### Per-Gesture Sample Counts

| Gesture | Target Samples | Rationale |
|---|---|---|
| SOS | 500+ | Primary emergency signal — highest accuracy required |
| EMERGENCY | 400+ | Critical alert — must distinguish from peace/scissors |
| ACCIDENT | 400+ | Fist — easily confused with relaxed hand |
| MEDICAL | 350+ | 4-finger — distinguish from SOS (5-finger) |
| SAFE | 300+ | Thumbs up — lower priority, good baseline |
| NONE | 800+ | Negative class — diverse non-gesture frames |

**Total minimum**: ~2,750 labeled samples across all classes.

---

## 2. Data Augmentation Pipeline

### Geometric Augmentations
- **Random rotation**: ±15° (simulates wrist angle variance)
- **Horizontal flip**: doubles effective dataset (left/right hand)
- **Random crop + resize**: 90–100% of original frame
- **Perspective warp**: subtle keystone distortion

### Photometric Augmentations
- **Brightness jitter**: ±30% (simulates lighting variance)
- **Contrast adjustment**: 0.7–1.3× factor
- **Gaussian noise injection**: σ = 5–15 (simulates sensor noise)
- **Color channel shuffle**: RGB → BGR permutations

### Temporal Augmentations (for sequence models)
- **Frame drop**: randomly remove 10–20% of frames in a sequence
- **Speed variation**: 0.8–1.2× playback speed
- **Sequence reversal**: mirror gesture entry/exit patterns

---

## 3. Evaluation Metrics

### Primary Metrics

| Metric | Target | Description |
|---|---|---|
| **Accuracy** | ≥ 95% | Overall correct predictions / total |
| **Precision** | ≥ 93% per class | True positives / (true + false positives) |
| **Recall** | ≥ 95% per class | True positives / (true + false negatives) |
| **F1-Score** | ≥ 94% macro-avg | Harmonic mean of precision and recall |

### Safety-Critical Metrics

| Metric | Target | Rationale |
|---|---|---|
| **SOS Recall** | ≥ 99% | A missed SOS is a life-safety failure |
| **False Positive Rate** | ≤ 2% | Spurious CRITICAL alerts erode trust |
| **Latency (P95)** | ≤ 100ms | Real-time requirement for edge deployment |

### Confusion Matrix Template

```
              Predicted
              SOS  EMR  ACC  MED  SAFE  NONE
Actual SOS  [ TP   .    .    .    .     FN  ]
       EMR  [  .   TP   .    .    .     FN  ]
       ACC  [  .    .   TP   .    .     FN  ]
       MED  [  .    .    .   TP   .     FN  ]
       SAFE [  .    .    .    .   TP    FN  ]
       NONE [ FP   FP   FP   FP   FP   TN  ]
```

---

## 4. Benchmark Protocol

### Cross-Validation
- **5-fold stratified cross-validation** on the full dataset
- Report mean ± std for all metrics

### Train/Test Split
- 80% train / 10% validation / 10% hold-out test
- Stratified by gesture label AND lighting condition

### Real-World Stress Test
- Test on 3 users **not** in the training set
- Test under **2 lighting conditions** not in training
- Report per-user accuracy breakdown

---

## 5. Model Comparison Framework

| Model | Accuracy | F1 | Latency | FPS | Notes |
|---|---|---|---|---|---|
| Random Forest (v1) | ~88% | ~85% | 2ms | 60+ | No temporal context |
| Physics Heuristic (v2) | 100%* | 100%* | <1ms | 60+ | *Deterministic, no learning |
| CNN (future) | TBD | TBD | ~15ms | 30+ | Requires GPU for training |
| CNN+LSTM (future) | TBD | TBD | ~25ms | 20+ | Best temporal accuracy |

*Physics heuristic is mathematically deterministic given correct landmarks — accuracy is bounded by MediaPipe's landmark detection quality.

---

## 6. Privacy-Aware Data Handling

- **No raw face storage**: All training frames are anonymised (face regions blurred) before persistence
- **Landmark-only datasets**: Gesture training uses only 21×3 numeric landmark coordinates, never pixel data
- **Consent protocol**: All data collection requires explicit written consent with opt-out capability
- **Data retention**: Training data retained for 12 months, then securely deleted
- **GDPR compliance**: Right to erasure supported — individual subjects can request removal
