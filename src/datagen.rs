//! Synthetic Facility Data Generator
//!
//! Generates realistic facility health snapshots calibrated against
//! real NexusEdge deployments (Warren, FCOG, Huntington).
//!
//! Four conditions: Normal, Watch, Warning, Critical
//! Each maps to a distinct feature distribution and health label range.

use crate::config::NUM_FEATURES;

// =============================================================================
// Seeded RNG
// =============================================================================

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }

    pub fn gauss(&mut self, mean: f32, std: f32) -> f32 {
        let u1 = self.next_f32().max(1e-7);
        let u2 = self.next_f32();
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
        mean + z * std
    }

    pub fn gauss_clamped(&mut self, mean: f32, std: f32, lo: f32, hi: f32) -> f32 {
        self.gauss(mean, std).clamp(lo, hi)
    }
}

// =============================================================================
// Conditions
// =============================================================================

#[derive(Clone, Copy, Debug)]
pub enum Condition {
    Normal,
    Watch,
    Warning,
    Critical,
}

impl Condition {
    /// Health label range for this condition.
    pub fn health_label(self, rng: &mut Rng) -> f32 {
        match self {
            Self::Normal => rng.gauss_clamped(0.92, 0.04, 0.82, 1.0),
            Self::Watch => rng.gauss_clamped(0.70, 0.06, 0.55, 0.80),
            Self::Warning => rng.gauss_clamped(0.40, 0.08, 0.22, 0.55),
            Self::Critical => rng.gauss_clamped(0.12, 0.06, 0.0, 0.22),
        }
    }

    /// Severity class index (0=normal, 1=watch, 2=warning, 3=critical).
    pub fn severity_class(self) -> usize {
        match self {
            Self::Normal => 0,
            Self::Watch => 1,
            Self::Warning => 2,
            Self::Critical => 3,
        }
    }
}

// =============================================================================
// Sample Generation
// =============================================================================

/// Generate a single facility snapshot with (features, health_label, severity_class).
pub fn generate_sample(rng: &mut Rng, condition: Condition) -> (Vec<f32>, f32, usize) {
    let mut f = vec![0.0f32; NUM_FEATURES];

    match condition {
        Condition::Normal => fill_normal(rng, &mut f),
        Condition::Watch => fill_watch(rng, &mut f),
        Condition::Warning => fill_warning(rng, &mut f),
        Condition::Critical => fill_critical(rng, &mut f),
    }

    // Temporal features (same for all conditions)
    f[42] = rng.next_f32(); // hour_of_day
    f[43] = rng.next_f32(); // day_of_week
    f[44] = if rng.next_f32() > 0.3 { 1.0 } else { 0.0 }; // is_occupied

    let label = condition.health_label(rng);
    let severity = condition.severity_class();

    (f, label, severity)
}

fn fill_normal(rng: &mut Rng, f: &mut [f32]) {
    // Equipment overview
    f[0] = rng.gauss_clamped(0.50, 0.18, 0.15, 0.85);
    f[1] = rng.gauss_clamped(0.95, 0.04, 0.85, 1.0);
    f[2] = rng.gauss_clamped(0.30, 0.10, 0.10, 0.60);
    f[3] = rng.gauss_clamped(0.02, 0.02, 0.0, 0.10);

    // Anomaly: low
    f[4] = rng.gauss_clamped(0.10, 0.04, 0.02, 0.20);
    f[5] = rng.gauss_clamped(0.18, 0.05, 0.05, 0.28);
    f[6] = rng.gauss_clamped(0.04, 0.02, 0.01, 0.12);
    f[7] = 0.0;
    f[8] = rng.gauss_clamped(0.0, 0.01, -0.05, 0.05);
    f[9] = rng.gauss_clamped(0.0, 0.005, -0.02, 0.02);

    // FDD: baseline noise (real models output 15-50% confidence as normal)
    f[10] = rng.gauss_clamped(0.28, 0.10, 0.10, 0.50);
    f[11] = rng.gauss_clamped(0.30, 0.10, 0.10, 0.50);
    f[12] = rng.gauss_clamped(0.32, 0.10, 0.10, 0.55);
    f[13] = rng.gauss_clamped(0.30, 0.15, 0.0, 0.55);
    f[14] = rng.gauss_clamped(0.30, 0.15, 0.0, 0.55);
    f[15] = rng.gauss_clamped(0.30, 0.15, 0.0, 0.55);
    f[16] = rng.gauss_clamped(0.35, 0.15, 0.1, 0.6);
    f[17] = rng.gauss_clamped(0.0, 0.01, -0.05, 0.05);

    // Alarms: none
    f[18] = 0.0;
    f[19] = 0.0;
    f[20] = 0.0;
    f[21] = 0.0;

    // Temps: normal with natural variance from mixed equipment
    f[22] = rng.gauss_clamped(0.55, 0.12, 0.20, 0.85);
    f[23] = rng.gauss_clamped(0.20, 0.10, 0.05, 0.40);
    f[24] = rng.gauss_clamped(0.55, 0.12, 0.20, 0.85);
    f[25] = rng.gauss_clamped(0.20, 0.10, 0.05, 0.40);
    f[26] = rng.gauss_clamped(0.44, 0.06, 0.25, 0.65);
    f[27] = rng.gauss_clamped(0.07, 0.03, 0.01, 0.15);
    f[28] = rng.gauss_clamped(0.10, 0.08, 0.0, 0.35);
    f[29] = rng.gauss_clamped(0.06, 0.04, 0.0, 0.20);
    f[30] = 0.0; // no OOR zones
    f[31] = rng.gauss_clamped(0.02, 0.02, 0.0, 0.08);

    // Valves: normal positions
    f[32] = rng.gauss_clamped(0.30, 0.15, 0.0, 0.70);
    f[33] = rng.gauss_clamped(0.10, 0.10, 0.0, 0.40);
    f[34] = rng.gauss_clamped(0.05, 0.04, 0.0, 0.15);
    f[35] = rng.gauss_clamped(0.20, 0.15, 0.0, 0.60);

    // Fans
    f[36] = rng.gauss_clamped(0.50, 0.15, 0.25, 0.90);
    f[37] = 0.0;
    f[38] = rng.gauss_clamped(0.60, 0.15, 0.30, 0.90);

    // Environmental
    f[39] = rng.gauss_clamped(0.45, 0.20, 0.0, 1.0);
    f[40] = rng.gauss_clamped(0.50, 0.15, 0.0, 1.0);
    f[41] = rng.gauss_clamped(0.50, 0.20, 0.0, 1.0);

    // Energy: normal
    f[45] = rng.gauss_clamped(0.40, 0.15, 0.10, 0.70);
    f[46] = rng.gauss_clamped(0.75, 0.10, 0.50, 0.95);
    f[47] = rng.gauss_clamped(0.0, 0.01, -0.05, 0.05);
}

fn fill_watch(rng: &mut Rng, f: &mut [f32]) {
    fill_normal(rng, f);
    // Slightly elevated anomaly
    f[4] = rng.gauss_clamped(0.22, 0.06, 0.12, 0.35);
    f[5] = rng.gauss_clamped(0.32, 0.08, 0.18, 0.48);
    f[7] = rng.gauss_clamped(0.10, 0.08, 0.0, 0.25);
    f[8] = rng.gauss_clamped(0.03, 0.02, -0.01, 0.10);
    // Slightly higher FDD
    f[10] = rng.gauss_clamped(0.38, 0.10, 0.20, 0.58);
    f[13] = rng.gauss_clamped(0.40, 0.12, 0.20, 0.65);
    // A few alarms
    f[18] = rng.gauss_clamped(0.08, 0.06, 0.0, 0.20);
    // Slight temp drift
    f[30] = rng.gauss_clamped(0.05, 0.04, 0.0, 0.15);
    f[31] = rng.gauss_clamped(0.06, 0.04, 0.0, 0.15);
    // Energy slightly up
    f[45] = rng.gauss_clamped(0.50, 0.12, 0.25, 0.75);
    f[47] = rng.gauss_clamped(0.03, 0.02, -0.01, 0.08);
}

fn fill_warning(rng: &mut Rng, f: &mut [f32]) {
    fill_normal(rng, f);
    // Elevated anomaly
    f[4] = rng.gauss_clamped(0.35, 0.08, 0.20, 0.55);
    f[5] = rng.gauss_clamped(0.52, 0.10, 0.35, 0.70);
    f[6] = rng.gauss_clamped(0.12, 0.04, 0.05, 0.22);
    f[7] = rng.gauss_clamped(0.30, 0.12, 0.10, 0.55);
    f[8] = rng.gauss_clamped(0.08, 0.04, 0.0, 0.18);
    f[9] = rng.gauss_clamped(0.03, 0.02, 0.0, 0.08);
    // High FDD
    f[10] = rng.gauss_clamped(0.52, 0.12, 0.30, 0.75);
    f[13] = rng.gauss_clamped(0.55, 0.15, 0.30, 0.80);
    f[16] = rng.gauss_clamped(0.50, 0.12, 0.25, 0.75);
    f[17] = rng.gauss_clamped(0.06, 0.04, 0.0, 0.15);
    // Alarms
    f[18] = rng.gauss_clamped(0.20, 0.10, 0.05, 0.45);
    f[19] = rng.gauss_clamped(0.08, 0.06, 0.0, 0.22);
    f[20] = rng.gauss_clamped(0.15, 0.08, 0.02, 0.35);
    f[21] = rng.gauss_clamped(0.05, 0.04, 0.0, 0.15);
    // Temps drifting
    f[30] = rng.gauss_clamped(0.15, 0.08, 0.03, 0.35);
    f[31] = rng.gauss_clamped(0.12, 0.06, 0.02, 0.28);
    // Some valves saturated
    f[34] = rng.gauss_clamped(0.15, 0.08, 0.03, 0.30);
    // Fans issues
    f[37] = rng.gauss_clamped(0.08, 0.06, 0.0, 0.20);
    // Energy up
    f[45] = rng.gauss_clamped(0.60, 0.12, 0.35, 0.85);
    f[46] = rng.gauss_clamped(0.55, 0.12, 0.30, 0.75);
    f[47] = rng.gauss_clamped(0.06, 0.03, 0.0, 0.15);
}

fn fill_critical(rng: &mut Rng, f: &mut [f32]) {
    fill_normal(rng, f);
    // Very high anomaly
    f[4] = rng.gauss_clamped(0.55, 0.12, 0.35, 0.90);
    f[5] = rng.gauss_clamped(0.80, 0.10, 0.55, 1.0);
    f[6] = rng.gauss_clamped(0.18, 0.06, 0.06, 0.35);
    f[7] = rng.gauss_clamped(0.65, 0.15, 0.35, 1.0);
    f[8] = rng.gauss_clamped(0.15, 0.06, 0.03, 0.30);
    f[9] = rng.gauss_clamped(0.08, 0.04, 0.01, 0.18);
    // Very high FDD
    f[10] = rng.gauss_clamped(0.70, 0.12, 0.45, 0.95);
    f[11] = rng.gauss_clamped(0.65, 0.12, 0.40, 0.90);
    f[12] = rng.gauss_clamped(0.60, 0.10, 0.35, 0.85);
    f[13] = rng.gauss_clamped(0.70, 0.15, 0.40, 1.0);
    f[14] = rng.gauss_clamped(0.65, 0.15, 0.35, 0.95);
    f[15] = rng.gauss_clamped(0.60, 0.12, 0.30, 0.90);
    f[16] = rng.gauss_clamped(0.60, 0.15, 0.25, 0.90);
    f[17] = rng.gauss_clamped(0.12, 0.05, 0.02, 0.25);
    // Many alarms
    f[18] = rng.gauss_clamped(0.55, 0.15, 0.25, 0.90);
    f[19] = rng.gauss_clamped(0.35, 0.15, 0.10, 0.70);
    f[20] = rng.gauss_clamped(0.40, 0.15, 0.15, 0.75);
    f[21] = rng.gauss_clamped(0.20, 0.10, 0.05, 0.40);
    // Temps way off
    f[26] = rng.gauss_clamped(0.50, 0.15, 0.10, 0.90);
    f[27] = rng.gauss_clamped(0.18, 0.06, 0.05, 0.35);
    f[30] = rng.gauss_clamped(0.40, 0.15, 0.15, 0.75);
    f[31] = rng.gauss_clamped(0.25, 0.10, 0.08, 0.50);
    // Valves saturated
    f[34] = rng.gauss_clamped(0.35, 0.15, 0.10, 0.65);
    // Fans struggling
    f[36] = rng.gauss_clamped(0.30, 0.15, 0.05, 0.60);
    f[37] = rng.gauss_clamped(0.20, 0.12, 0.03, 0.45);
    // Data going stale
    f[1] = rng.gauss_clamped(0.75, 0.12, 0.50, 0.95);
    f[3] = rng.gauss_clamped(0.25, 0.15, 0.05, 0.60);
    // Energy spike
    f[45] = rng.gauss_clamped(0.80, 0.10, 0.55, 1.0);
    f[46] = rng.gauss_clamped(0.35, 0.12, 0.15, 0.55);
    f[47] = rng.gauss_clamped(0.15, 0.06, 0.03, 0.30);
}

/// Generate a batch of samples.
/// Returns (features [batch*48], health_labels [batch], severity_classes [batch]).
pub fn generate_batch(rng: &mut Rng, batch_size: usize) -> (Vec<f32>, Vec<f32>, Vec<usize>) {
    let mut all_features = Vec::with_capacity(batch_size * NUM_FEATURES);
    let mut all_labels = Vec::with_capacity(batch_size);
    let mut all_severity = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        // 40% normal, 25% watch, 20% warning, 15% critical
        let condition = match i % 20 {
            0..=7 => Condition::Normal,
            8..=12 => Condition::Watch,
            13..=16 => Condition::Warning,
            _ => Condition::Critical,
        };

        let (features, label, severity) = generate_sample(rng, condition);
        all_features.extend_from_slice(&features);
        all_labels.push(label);
        all_severity.push(severity);
    }

    (all_features, all_labels, all_severity)
}
