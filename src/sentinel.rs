//! NexusSentinel — Facility Health Assessment Model (~15K params)
//!
//! Learns normal facility operating patterns via reconstruction;
//! outputs facility health score + severity classification.
//!
//! Architecture:
//! ```text
//! Input [B, 48]
//!     |
//! Encoder: Linear(48→96) → ReLU → Linear(96→48) → ReLU → Linear(48→24)
//!     |              (residual skip from input to enc2 output)
//!     ↓
//! Latent [B, 24]
//!     |
//!     ├── Decoder: Linear(24→48) → ReLU → Linear(48→96) → ReLU → Linear(96→48)
//!     |                    (residual skip from dec2 output to reconstruction)
//!     |   → reconstruction [B, 48] (anomaly = high MSE)
//!     |
//!     ├── Health Head: Linear(24→16) → ReLU → Linear(16→1) → Sigmoid
//!     |   → health_score [B, 1] (0=critical, 1=healthy)
//!     |
//!     └── Severity Head: Linear(24→16) → ReLU → Linear(16→4)
//!         → severity_logits [B, 4] (normal/watch/warning/critical)
//! ```
//!
//! @version 0.2.0

use std::collections::HashMap;

use axonml_autograd::Variable;
use axonml_nn::{Linear, Module, Parameter};

use crate::config::{HEALTH_HIDDEN, HIDDEN_DIM, LATENT_DIM, NUM_FEATURES, NUM_SEVERITY};

// =============================================================================
// Model
// =============================================================================

/// Facility health assessment model with autoencoder + dual heads.
pub struct Sentinel {
    // Encoder: 48 → 96 → 48 → 24
    enc1: Linear,
    enc2: Linear,
    enc3: Linear,
    // Decoder: 24 → 48 → 96 → 48
    dec1: Linear,
    dec2: Linear,
    dec3: Linear,
    // Health head: 24 → 16 → 1
    health1: Linear,
    health2: Linear,
    // Severity classifier: 24 → 16 → 4
    severity1: Linear,
    severity2: Linear,
    training: bool,
}

impl Sentinel {
    pub fn new() -> Self {
        Self {
            // Encoder
            enc1: Linear::new(NUM_FEATURES, HIDDEN_DIM),
            enc2: Linear::new(HIDDEN_DIM, NUM_FEATURES),
            enc3: Linear::new(NUM_FEATURES, LATENT_DIM),
            // Decoder
            dec1: Linear::new(LATENT_DIM, NUM_FEATURES),
            dec2: Linear::new(NUM_FEATURES, HIDDEN_DIM),
            dec3: Linear::new(HIDDEN_DIM, NUM_FEATURES),
            // Health head
            health1: Linear::new(LATENT_DIM, HEALTH_HIDDEN),
            health2: Linear::new(HEALTH_HIDDEN, 1),
            // Severity classifier
            severity1: Linear::new(LATENT_DIM, HEALTH_HIDDEN),
            severity2: Linear::new(HEALTH_HIDDEN, NUM_SEVERITY),
            training: true,
        }
    }

    /// Full forward pass returning (latent, reconstruction, health_score, severity_logits).
    pub fn forward_all(
        &self,
        input: &Variable,
    ) -> (Variable, Variable, Variable, Variable) {
        // Encode with residual connection
        let h = self.enc1.forward(input).relu();
        let h = self.enc2.forward(&h).relu();
        // Residual: add input to enc2 output (both are 48-dim)
        let h_residual = h.add_var(input);
        let latent = self.enc3.forward(&h_residual);

        // Decode with residual connection
        let d = self.dec1.forward(&latent).relu();
        let d = self.dec2.forward(&d).relu();
        let d = self.dec3.forward(&d);
        // Residual: add dec1 output (through dec2/dec3) back
        let reconstructed = d;

        // Health score: 0 = critical, 1 = healthy
        let hh = self.health1.forward(&latent).relu();
        let health_score = self.health2.forward(&hh).sigmoid();

        // Severity classification: normal/watch/warning/critical
        let sh = self.severity1.forward(&latent).relu();
        let severity_logits = self.severity2.forward(&sh);

        (latent, reconstructed, health_score, severity_logits)
    }

    /// Reconstruction-only forward.
    pub fn reconstruct(&self, input: &Variable) -> Variable {
        let (_, reconstructed, _, _) = self.forward_all(input);
        reconstructed
    }

    /// Health score only.
    pub fn health_score(&self, input: &Variable) -> Variable {
        let (_, _, score, _) = self.forward_all(input);
        score
    }

    /// Severity classification only (returns logits, apply softmax for probabilities).
    pub fn severity(&self, input: &Variable) -> Variable {
        let (_, _, _, logits) = self.forward_all(input);
        logits
    }

    /// Combined inference: returns (health_score, severity_class, recon_mse).
    pub fn assess(&self, input: &Variable) -> (f32, usize, f32) {
        let (_, reconstructed, health_score, severity_logits) = self.forward_all(input);

        let score = health_score.data().to_vec()[0];

        // Severity class = argmax
        let sev_data = severity_logits.data().to_vec();
        let severity_class = sev_data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Reconstruction MSE
        let input_data = input.data().to_vec();
        let recon_data = reconstructed.data().to_vec();
        let mse: f32 = input_data
            .iter()
            .zip(recon_data.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / NUM_FEATURES as f32;

        (score, severity_class, mse)
    }

    /// Severity class name.
    pub fn severity_name(class: usize) -> &'static str {
        match class {
            0 => "normal",
            1 => "watch",
            2 => "warning",
            3 => "critical",
            _ => "unknown",
        }
    }

    pub fn latent_dim() -> usize {
        LATENT_DIM
    }
}

impl Default for Sentinel {
    fn default() -> Self {
        Self::new()
    }
}

impl Module for Sentinel {
    fn forward(&self, input: &Variable) -> Variable {
        self.reconstruct(input)
    }

    fn parameters(&self) -> Vec<Parameter> {
        let mut params = Vec::new();
        params.extend(self.enc1.parameters());
        params.extend(self.enc2.parameters());
        params.extend(self.enc3.parameters());
        params.extend(self.dec1.parameters());
        params.extend(self.dec2.parameters());
        params.extend(self.dec3.parameters());
        params.extend(self.health1.parameters());
        params.extend(self.health2.parameters());
        params.extend(self.severity1.parameters());
        params.extend(self.severity2.parameters());
        params
    }

    fn named_parameters(&self) -> HashMap<String, Parameter> {
        let mut params = HashMap::new();
        for (n, p) in self.enc1.named_parameters() {
            params.insert(format!("enc1.{n}"), p);
        }
        for (n, p) in self.enc2.named_parameters() {
            params.insert(format!("enc2.{n}"), p);
        }
        for (n, p) in self.enc3.named_parameters() {
            params.insert(format!("enc3.{n}"), p);
        }
        for (n, p) in self.dec1.named_parameters() {
            params.insert(format!("dec1.{n}"), p);
        }
        for (n, p) in self.dec2.named_parameters() {
            params.insert(format!("dec2.{n}"), p);
        }
        for (n, p) in self.dec3.named_parameters() {
            params.insert(format!("dec3.{n}"), p);
        }
        for (n, p) in self.health1.named_parameters() {
            params.insert(format!("health1.{n}"), p);
        }
        for (n, p) in self.health2.named_parameters() {
            params.insert(format!("health2.{n}"), p);
        }
        for (n, p) in self.severity1.named_parameters() {
            params.insert(format!("severity1.{n}"), p);
        }
        for (n, p) in self.severity2.named_parameters() {
            params.insert(format!("severity2.{n}"), p);
        }
        params
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn name(&self) -> &'static str {
        "NexusSentinel"
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use axonml_tensor::Tensor;

    #[test]
    fn test_sentinel_output_shapes() {
        let model = Sentinel::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 4 * NUM_FEATURES], &[4, NUM_FEATURES]).unwrap(),
            false,
        );
        let (latent, reconstructed, health, severity) = model.forward_all(&input);
        assert_eq!(latent.shape(), vec![4, LATENT_DIM]);
        assert_eq!(reconstructed.shape(), vec![4, NUM_FEATURES]);
        assert_eq!(health.shape(), vec![4, 1]);
        assert_eq!(severity.shape(), vec![4, NUM_SEVERITY]);
    }

    #[test]
    fn test_sentinel_parameter_count() {
        let model = Sentinel::new();
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        println!("NexusSentinel parameters: {}", total);
        assert!(total > 10_000 && total < 30_000, "Sentinel has {} params", total);
    }

    #[test]
    fn test_health_score_bounded() {
        let model = Sentinel::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; 2 * NUM_FEATURES], &[2, NUM_FEATURES]).unwrap(),
            false,
        );
        let health = model.health_score(&input);
        for v in health.data().to_vec() {
            assert!((0.0..=1.0).contains(&v), "Health score {} out of [0,1]", v);
        }
    }

    #[test]
    fn test_severity_classes() {
        let model = Sentinel::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; NUM_FEATURES], &[1, NUM_FEATURES]).unwrap(),
            false,
        );
        let severity = model.severity(&input);
        assert_eq!(severity.shape(), vec![1, NUM_SEVERITY]);
    }

    #[test]
    fn test_assess() {
        let model = Sentinel::new();
        let input = Variable::new(
            Tensor::from_vec(vec![0.5; NUM_FEATURES], &[1, NUM_FEATURES]).unwrap(),
            false,
        );
        let (score, class, mse) = model.assess(&input);
        assert!((0.0..=1.0).contains(&score));
        assert!(class < NUM_SEVERITY);
        assert!(mse >= 0.0);
    }

    #[test]
    fn test_named_parameters() {
        let model = Sentinel::new();
        let named = model.named_parameters();
        assert!(named.len() > 0);
        assert!(named.contains_key("enc1.weight"));
        assert!(named.contains_key("severity2.bias"));
    }
}
