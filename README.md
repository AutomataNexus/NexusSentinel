<p align="center">
  <img src="assets/NexusSentinel_logo.PNG" alt="NexusSentinel Logo" width="280"/>
</p>

<h1 align="center">NexusSentinel</h1>

<p align="center">
  <strong>Facility Health Assessment for Edge Deployment</strong>
</p>

<p align="center">
  <a href="https://github.com/AutomataNexus/NexusSentinel/actions/workflows/ci.yml"><img src="https://github.com/AutomataNexus/NexusSentinel/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://crates.io/crates/axonml"><img src="https://img.shields.io/badge/built%20with-AxonML%200.4.3-blue" alt="AxonML"></a>
  <a href="https://github.com/AutomataNexus/NexusSentinel/releases"><img src="https://img.shields.io/github/v/release/AutomataNexus/NexusSentinel?include_prereleases" alt="Release"></a>
  <img src="https://img.shields.io/badge/SLSA-Level%203-green" alt="SLSA Level 3">
  <img src="https://img.shields.io/badge/params-~15K-orange" alt="Parameters">
  <img src="https://img.shields.io/badge/rust-edition%202024-red" alt="Rust Edition 2024">
  <a href="LICENSE-MIT"><img src="https://img.shields.io/badge/license-MIT%2FApache--2.0-lightgrey" alt="License"></a>
</p>

<p align="center">
  An MLP autoencoder with health scoring and severity classification for real-time facility health monitoring on edge devices. Takes a 48-dimensional feature vector aggregated from all equipment inference signals and outputs health score, severity class, and anomaly detection via reconstruction error.
</p>

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Feature Vector (48-dim)](#feature-vector-48-dim)
- [Training](#training)
- [Inference](#inference)
- [Edge Deployment](#edge-deployment)
- [Validation Results](#validation-results)
- [SLSA Provenance](#slsa-provenance)
- [Configuration Reference](#configuration-reference)
- [Project Structure](#project-structure)

---

## Overview

NexusSentinel is an edge-deployable neural network that assesses facility health in real time. It sits on top of per-equipment inference models (anomaly detection, fault detection & diagnostics, predictive maintenance) and aggregates their outputs into a single facility-wide health assessment.

### What it does

| Input | Output | Use Case |
|-------|--------|----------|
| 48-dim facility feature vector | **Health score** (0-1) | Dashboard display, alerting thresholds |
| (aggregated from all equipment) | **Severity class** (normal/watch/warning/critical) | Automated escalation routing |
| | **Reconstruction error** (anomaly signal) | Novel pattern detection |

### Why it exists

Individual equipment models produce hundreds of signals per facility — anomaly scores, fault predictions, confidence levels, trend data. Building operators and facility managers need a **single number** that answers: "Is my building healthy right now?" NexusSentinel distills all those signals into one actionable assessment.

### Design constraints

- **~15K parameters** — runs on any edge device (Raspberry Pi, NexusEdge controller, cloud VM)
- **<1ms inference** — real-time health assessment at any polling interval
- **No GPU required** — pure CPU inference for edge deployment
- **JSON weight export** — TypeScript runtime compatible for web-based controllers

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NexusSentinel v0.2                        │
│                    ~15K parameters                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input [B, 48]  (facility feature vector)                  │
│       │                                                     │
│  ┌────┴──────────────────────────────────┐                 │
│  │ ENCODER                                │                 │
│  │                                        │                 │
│  │  Linear(48 → 96) + ReLU              │                 │
│  │  Linear(96 → 48) + ReLU              │                 │
│  │       + residual ←─── input           │  ← skip conn   │
│  │  Linear(48 → 24)                     │                 │
│  │                                        │                 │
│  │  Output: latent [B, 24]              │                 │
│  └────┬──────────┬──────────┬────────────┘                 │
│       │          │          │                               │
│  ┌────┴────┐ ┌───┴────┐ ┌──┴──────────┐                   │
│  │ DECODER │ │ HEALTH │ │  SEVERITY   │                   │
│  │         │ │  HEAD  │ │ CLASSIFIER  │                   │
│  │ 24→48   │ │ 24→16  │ │  24→16     │                   │
│  │ 48→96   │ │  ReLU  │ │   ReLU     │                   │
│  │ 96→48   │ │ 16→1   │ │  16→4      │                   │
│  │         │ │ sigmoid │ │            │                   │
│  └────┬────┘ └───┬────┘ └──┬──────────┘                   │
│       │          │          │                               │
│  reconstruct  health     severity                          │
│  [B, 48]     [B, 1]     [B, 4]                            │
│   (MSE →      (0=dead    (normal/watch/                    │
│   anomaly)    1=healthy)  warning/critical)                │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Parameter Breakdown

| Component | Shape | Parameters | Purpose |
|-----------|-------|-----------|---------|
| **Encoder 1** | 48→96 | 4,704 | Expand to hidden dim |
| **Encoder 2** | 96→48 | 4,656 | Compress back (+ residual from input) |
| **Encoder 3** | 48→24 | 1,176 | Bottleneck to latent |
| **Decoder 1** | 24→48 | 1,200 | Expand from latent |
| **Decoder 2** | 48→96 | 4,704 | Hidden dim |
| **Decoder 3** | 96→48 | 4,656 | Reconstruct input |
| **Health 1** | 24→16 | 400 | Health feature extraction |
| **Health 2** | 16→1 | 17 | Health score output |
| **Severity 1** | 24→16 | 400 | Severity feature extraction |
| **Severity 2** | 16→4 | 68 | 4-class classification |
| **Total** | | **~15,981** | |

### Key Design Decisions

- **Residual skip connection** — encoder layer 2 output adds the raw input, preventing information loss through the bottleneck and helping gradients flow during training
- **Separate heads** — health score (regression) and severity (classification) use independent pathways from the latent space, preventing task interference
- **Reconstruction-based anomaly** — the autoencoder learns "normal" patterns; novel/abnormal inputs reconstruct poorly (high MSE), providing an unsupervised anomaly signal without labeled anomaly data
- **4-class severity vs 3-class** — added "watch" tier between normal and warning for earlier intervention (real facilities often have a "keep an eye on it" state)

---

## Feature Vector (48-dim)

All features normalized to [0, 1]. Calibrated against real NexusEdge deployments (Warren 62-equip, FCOG 12-equip, Huntington).

### Equipment Overview (0-3)

| Index | Feature | Normal Range | Description |
|-------|---------|-------------|-------------|
| 0 | `num_equipment` | 0.15-0.85 | Total equipment count (normalized) |
| 1 | `pct_equipment_online` | 0.85-1.0 | % of equipment with fresh data |
| 2 | `equipment_diversity` | 0.10-0.60 | Distinct equipment types (AHU, FCU, boiler...) |
| 3 | `data_freshness` | 0.0-0.10 | Age of most recent data (0=fresh, 1=stale) |

### Anomaly Detection (4-9)

| Index | Feature | Normal | Watch | Warning | Critical |
|-------|---------|--------|-------|---------|----------|
| 4 | `mean_anomaly_score` | 0.02-0.20 | 0.12-0.35 | 0.20-0.55 | 0.35-0.90 |
| 5 | `max_anomaly_score` | 0.05-0.28 | 0.18-0.48 | 0.35-0.70 | 0.55-1.0 |
| 6 | `std_anomaly_score` | 0.01-0.12 | — | 0.05-0.22 | 0.06-0.35 |
| 7 | `pct_anomaly_flagged` | 0.0 | 0.0-0.25 | 0.10-0.55 | 0.35-1.0 |
| 8 | `anomaly_trend_slope` | ±0.05 | 0.0-0.10 | 0.0-0.18 | 0.03-0.30 |
| 9 | `anomaly_acceleration` | ±0.02 | — | 0.0-0.08 | 0.01-0.18 |

### FDD Predictions (10-17)

| Index | Feature | Description |
|-------|---------|-------------|
| 10-12 | `mean_fdd_confidence_{5,15,30}min` | Avg confidence of top fault prediction at each horizon |
| 13-15 | `pct_fdd_failure_{5,15,30}min` | % of equipment with is_failure=1 at each horizon |
| 16 | `num_distinct_faults` | Diversity of predicted fault types |
| 17 | `fdd_trend_slope` | Is failure confidence trending up? |

**Important:** Real FDD models output 15-50% confidence as baseline noise. Low-confidence "failure" flags (< 25%) on FCUs are normal behavior, not actual failures. NexusSentinel is calibrated to handle this.

### Alarms (18-21)

| Index | Feature | Normal | Critical |
|-------|---------|--------|----------|
| 18 | `num_active_alarms` | 0 | 0.25-0.90 |
| 19 | `num_critical_alarms` | 0 | 0.10-0.70 |
| 20 | `alarm_rate` | 0 | 0.15-0.75 |
| 21 | `alarm_escalation_rate` | 0 | 0.05-0.40 |

### Temperatures (22-31)

| Index | Feature | Description |
|-------|---------|-------------|
| 22-23 | `mean/std_supply_temp` | Supply air/water temperature distribution |
| 24-25 | `mean/std_return_temp` | Return temperature distribution |
| 26-27 | `mean/std_space_temp` | Zone temperatures (should be tight around setpoint) |
| 28-29 | `temp_delta_mean/std` | Supply-return differential |
| 30 | `pct_space_temp_oor` | % of zones out of setpoint range |
| 31 | `max_space_temp_deviation` | Worst zone deviation from setpoint |

**Note:** Mixed equipment types (steam bundles 150°F + FCUs 65°F) cause naturally high supply temp variance — this is normal, not a fault.

### Valves, Fans, Environment, Temporal, Energy (32-47)

| Range | Features |
|-------|----------|
| 32-35 | HW/CW valve positions, saturated %, damper position |
| 36-38 | Fan running %, commanded-off %, VFD speed |
| 39-41 | Outdoor temp, humidity, HVAC mode |
| 42-44 | Hour of day, day of week, occupancy flag |
| 45-47 | Energy usage, efficiency score, energy trend |

---

## Training

```bash
# Train (200 epochs, cosine LR, live browser monitor)
cargo run --release --bin train
```

### Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Epochs | 200 | Sufficient for ~15K param model |
| Batch size | 64 | Balanced gradient noise |
| Batches/epoch | 32 | 2,048 samples per epoch |
| Learning rate | 0.002 → 1e-6 | Cosine annealing schedule |
| Optimizer | Adam | Standard for small models |

### Loss Function

Three losses combined with weights:

```
L = 1.0 × MSE(reconstruction, input)     # Autoencoder reconstruction
  + 3.0 × MSE(health_score, label)        # Health score regression
  + 1.5 × CrossEntropy(severity, class)   # Severity classification
```

| Component | Weight | Final Loss | Purpose |
|-----------|--------|-----------|---------|
| Reconstruction | 1.0 | 0.008 | Learn normal operating patterns |
| Health | 3.0 | 0.005 | Accurate 0-1 health scoring |
| Severity | 1.5 | 0.002 | Correct 4-class classification |

### Synthetic Data Generation

Training uses synthetic facility snapshots calibrated to real deployments:

| Condition | Distribution | Health Label | Examples |
|-----------|-------------|--------------|----------|
| **Normal** (40%) | Low anomaly, no alarms, tight temps | 0.82-1.0 | Warren on a good day |
| **Watch** (25%) | Slightly elevated anomaly, minor FDD | 0.55-0.80 | One equipment drifting |
| **Warning** (20%) | High anomaly, alarms, temp drift | 0.22-0.55 | Multiple equipment issues |
| **Critical** (15%) | Widespread failure, many alarms | 0.0-0.22 | Facility-wide problems |

### Training Features

- **Live browser training monitor** (AxonML TrainingMonitor) — real-time loss curves
- **Checkpoint save every epoch** (`checkpoint_latest.axonml`)
- **Auto-resume on restart** — picks up from latest checkpoint
- **Best model tracking** — saves `checkpoint_best.axonml` on improvement
- **Periodic snapshots** every 20 epochs

---

## Inference

### Rust Binary

```bash
# From JSON file
cargo run --release --bin inference -- --input facility_snapshot.json

# From stdin
echo '{"features":[0.5,0.95,0.3,...48 values...]}' | cargo run --release --bin inference
```

**Output:**
```json
{
  "health_score": 0.9234,
  "severity": "normal",
  "severity_class": 0,
  "reconstruction_mse": 0.0023,
  "anomaly_detected": false
}
```

### Programmatic (Rust)

```rust
use nexus_sentinel::Sentinel;

let model = Sentinel::new();
// ... load weights ...

let (health_score, severity_class, recon_mse) = model.assess(&input);
println!("Health: {:.2}, Severity: {}", health_score, Sentinel::severity_name(severity_class));
```

---

## Edge Deployment

For TypeScript/JavaScript edge runtimes (NexusEdge controllers, web dashboards):

### Step 1: Export Weights

```bash
cargo run --release --bin export-weights
# → models/sentinel_weights.json (250 KB)
```

### Step 2: Load in TypeScript

```typescript
import weights from './sentinel_weights.json';

function forward(features: number[]): { health: number; severity: string } {
  // Encoder
  let h = relu(matmul(features, weights.enc1.weight, weights.enc1.bias));
  h = relu(matmul(h, weights.enc2.weight, weights.enc2.bias));
  // Residual: add input
  h = h.map((v, i) => v + features[i]);
  const latent = matmul(h, weights.enc3.weight, weights.enc3.bias);

  // Health head
  let hh = relu(matmul(latent, weights.health1.weight, weights.health1.bias));
  const health = sigmoid(matmul(hh, weights.health2.weight, weights.health2.bias)[0]);

  // Severity head
  let sh = relu(matmul(latent, weights.severity1.weight, weights.severity1.bias));
  const sevLogits = matmul(sh, weights.severity2.weight, weights.severity2.bias);
  const sevClass = sevLogits.indexOf(Math.max(...sevLogits));
  const sevNames = ['normal', 'watch', 'warning', 'critical'];

  return { health, severity: sevNames[sevClass] };
}
```

### Step 3: Deploy

The JSON weights file contains all matrices in row-major format. No ML library dependency needed — just matrix multiplication, ReLU, and sigmoid. Runs in any JavaScript runtime.

---

## Validation Results

Trained on synthetic data calibrated to real NexusEdge deployments:

```
Condition   │ Health Score │ Expected │ Severity   │ Recon MSE
────────────┼──────────────┼──────────┼────────────┼──────────
Normal      │ 0.9164       │ ~0.92    │ ✓ normal   │ 0.0047
Watch       │ 0.6717       │ ~0.70    │ ✓ watch    │ 0.0059
Warning     │ 0.3650       │ ~0.40    │ ✓ warning  │ 0.0105
Critical    │ 0.1299       │ ~0.12    │ ✓ critical │ 0.0154
```

**Key observations:**
- All 4 severity classes correctly predicted
- Health scores track expectations within ±0.05
- Reconstruction MSE monotonically increases with degradation (0.005 → 0.015)
- Model converges in ~44 seconds on CPU (200 epochs)

---

## SLSA Provenance

All releases include **SLSA Level 3** provenance attestations signed via Sigstore.

### Verify a release:

```bash
slsa-verifier verify-artifact \
  nexus-sentinel-train-0.2.0-linux-amd64 \
  --provenance-path multiple.intoto.jsonl \
  --source-uri github.com/AutomataNexus/NexusSentinel \
  --source-tag v0.2.0
```

### Release artifacts:

| Artifact | Description |
|----------|-------------|
| `nexus-sentinel-train-*-linux-amd64` | Training binary |
| `nexus-sentinel-inference-*-linux-amd64` | Inference binary |
| `nexus-sentinel-export-*-linux-amd64` | Weight export tool |
| `sentinel-*.axonml` | Trained model weights |
| `sentinel-*-weights.json` | JSON weights for TypeScript |
| `multiple.intoto.jsonl` | SLSA L3 provenance attestation |

---

## Configuration Reference

### Training CLI

```bash
cargo run --release --bin train
```

No CLI flags — configuration is in `src/bin/train.rs` constants:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_epochs` | 200 | Training epochs |
| `batch_size` | 64 | Samples per batch |
| `batches_per_epoch` | 32 | Batches per epoch (2,048 samples) |
| `learning_rate` | 0.002 | Initial learning rate |
| `recon_weight` | 1.0 | Reconstruction loss weight |
| `health_weight` | 3.0 | Health score loss weight |
| `severity_weight` | 1.5 | Severity classification loss weight |

### Model Constants (`src/config.rs`)

| Constant | Value | Description |
|----------|-------|-------------|
| `NUM_FEATURES` | 48 | Input feature vector dimension |
| `LATENT_DIM` | 24 | Autoencoder bottleneck dimension |
| `HIDDEN_DIM` | 96 | Encoder/decoder hidden dimension |
| `HEALTH_HIDDEN` | 16 | Head hidden dimension |
| `NUM_SEVERITY` | 4 | Severity classes |

---

## Project Structure

```
NexusSentinel/
├── .github/
│   └── workflows/
│       ├── ci.yml                  # CI: check, test, clippy, fmt
│       └── slsa-release.yml        # SLSA Level 3 provenance on tag push
├── assets/
│   └── NexusSentinel_logo.PNG      # Project logo
├── Cargo.toml                      # Rust edition 2024, AxonML 0.4.3
├── README.md                       # This file
├── src/
│   ├── lib.rs                      # Library root
│   ├── config.rs                   # 48-feature vector definition + model dimensions
│   ├── sentinel.rs                 # Model architecture (~15K params, 7 tests)
│   ├── datagen.rs                  # Synthetic facility data generator (4 conditions)
│   └── bin/
│       ├── train.rs                # Training: monitor + checkpoint/resume + cosine LR
│       ├── inference.rs            # JSON input → health assessment JSON output
│       └── export_weights.rs       # Export to JSON for TypeScript edge runtime
├── models/
│   ├── sentinel.axonml             # Trained model weights (binary)
│   └── sentinel_weights.json       # Exported weights for TypeScript (250 KB)
└── checkpoints/
    ├── checkpoint_latest.axonml    # Most recent (for resume)
    ├── checkpoint_best.axonml      # Best loss achieved
    └── checkpoint_epoch_NNNN.axonml # Periodic snapshots
```

---

## Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `axonml-core` | 0.4.3 | Device management |
| `axonml-tensor` | 0.4.3 | Tensor operations |
| `axonml-autograd` | 0.4.3 | Automatic differentiation |
| `axonml-nn` | 0.4.3 | Linear layers, CrossEntropyLoss, Module trait |
| `axonml-optim` | 0.4.3 | Adam optimizer, CosineAnnealingLR |
| `axonml-serialize` | 0.4.3 | Checkpoint save/load, state dict |
| `axonml` | 0.4.3 | Training monitor |

---

<p align="center">
  <sub>Built with <a href="https://github.com/AutomataNexus/AxonML">AxonML</a> — a Rust deep learning framework by <a href="https://automatanexus.com">AutomataNexus LLC</a></sub>
</p>
