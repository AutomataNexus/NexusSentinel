//! Export NexusSentinel weights to JSON for TypeScript edge runtime.
//!
//! Usage: cargo run --release --bin export-weights

use axonml_serialize::load_state_dict;
use nexus_sentinel::config::{HEALTH_HIDDEN, HIDDEN_DIM, LATENT_DIM, NUM_FEATURES, NUM_SEVERITY};
use serde::Serialize;

#[derive(Serialize)]
struct LinearWeights {
    weight: Vec<Vec<f32>>,
    bias: Vec<f32>,
    in_features: usize,
    out_features: usize,
}

#[derive(Serialize)]
struct SentinelWeights {
    version: String,
    enc1: LinearWeights,
    enc2: LinearWeights,
    enc3: LinearWeights,
    dec1: LinearWeights,
    dec2: LinearWeights,
    dec3: LinearWeights,
    health1: LinearWeights,
    health2: LinearWeights,
    severity1: LinearWeights,
    severity2: LinearWeights,
    num_features: usize,
    latent_dim: usize,
    num_severity: usize,
}

fn extract_linear(
    state_dict: &axonml_serialize::StateDict,
    prefix: &str,
    in_features: usize,
    out_features: usize,
) -> LinearWeights {
    let weight_key = format!("{prefix}.weight");
    let bias_key = format!("{prefix}.bias");

    let weight_entry = state_dict
        .get(&weight_key)
        .unwrap_or_else(|| panic!("Missing: {weight_key}"));
    let bias_entry = state_dict
        .get(&bias_key)
        .unwrap_or_else(|| panic!("Missing: {bias_key}"));

    let w_flat = weight_entry.data.to_tensor().unwrap().to_vec();
    let mut weight = Vec::with_capacity(out_features);
    for row in 0..out_features {
        let start = row * in_features;
        weight.push(w_flat[start..start + in_features].to_vec());
    }

    LinearWeights {
        weight,
        bias: bias_entry.data.to_tensor().unwrap().to_vec(),
        in_features,
        out_features,
    }
}

fn main() {
    let model_path = "/opt/NexusSentinel/models/sentinel.axonml";
    let output_path = "/opt/NexusSentinel/models/sentinel_weights.json";

    println!("=== NexusSentinel Weight Export ===");
    println!("Loading: {model_path}");

    let state_dict = load_state_dict(model_path).expect("Failed to load state dict");

    println!("Found {} entries:", state_dict.len());
    for (name, entry) in state_dict.entries() {
        println!("  {name} {:?}", entry.data.shape());
    }

    let weights = SentinelWeights {
        version: "0.2.0".to_string(),
        enc1: extract_linear(&state_dict, "enc1", NUM_FEATURES, HIDDEN_DIM),
        enc2: extract_linear(&state_dict, "enc2", HIDDEN_DIM, NUM_FEATURES),
        enc3: extract_linear(&state_dict, "enc3", NUM_FEATURES, LATENT_DIM),
        dec1: extract_linear(&state_dict, "dec1", LATENT_DIM, NUM_FEATURES),
        dec2: extract_linear(&state_dict, "dec2", NUM_FEATURES, HIDDEN_DIM),
        dec3: extract_linear(&state_dict, "dec3", HIDDEN_DIM, NUM_FEATURES),
        health1: extract_linear(&state_dict, "health1", LATENT_DIM, HEALTH_HIDDEN),
        health2: extract_linear(&state_dict, "health2", HEALTH_HIDDEN, 1),
        severity1: extract_linear(&state_dict, "severity1", LATENT_DIM, HEALTH_HIDDEN),
        severity2: extract_linear(&state_dict, "severity2", HEALTH_HIDDEN, NUM_SEVERITY),
        num_features: NUM_FEATURES,
        latent_dim: LATENT_DIM,
        num_severity: NUM_SEVERITY,
    };

    let json = serde_json::to_string(&weights).expect("Serialization failed");
    std::fs::write(output_path, &json).expect("Write failed");

    println!(
        "\nExported: {output_path} ({:.1} KB)",
        json.len() as f64 / 1024.0
    );
}
