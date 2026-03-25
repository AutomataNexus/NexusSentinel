//! NexusSentinel Inference — Single-shot facility health assessment
//!
//! Loads trained model and assesses a facility from a JSON feature vector.
//!
//! Usage:
//!   cargo run --release --bin inference -- --input features.json
//!   echo '{"features":[0.5,0.95,...]}' | cargo run --release --bin inference

use std::io::Read;
use std::path::PathBuf;

use axonml_autograd::Variable;
use axonml_nn::Module;
use axonml_serialize::load_state_dict;
use axonml_tensor::Tensor;

use nexus_sentinel::config::NUM_FEATURES;
use nexus_sentinel::Sentinel;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let model_path = args
        .iter()
        .position(|a| a == "--model")
        .map(|i| PathBuf::from(&args[i + 1]))
        .unwrap_or_else(|| PathBuf::from("/opt/NexusSentinel/models/sentinel.axonml"));

    // Load model
    let model = Sentinel::new();
    if model_path.exists() {
        let state_dict = load_state_dict(&model_path).expect("Failed to load model");
        let model_params = model.named_parameters();
        let mut loaded = 0;
        for (name, param) in &model_params {
            if let Some(entry) = state_dict.get(name) {
                if let Ok(tensor) = entry.data.to_tensor() {
                    if tensor.shape() == param.data().shape() {
                        param.update_data(tensor);
                        loaded += 1;
                    }
                }
            }
        }
        eprintln!("Loaded {}/{} parameters", loaded, model_params.len());
    } else {
        eprintln!("WARNING: No model at {}, using random weights", model_path.display());
    }

    // Read features from --input file or stdin
    let features_json = if let Some(pos) = args.iter().position(|a| a == "--input") {
        std::fs::read_to_string(&args[pos + 1]).expect("Failed to read input file")
    } else {
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf).expect("Failed to read stdin");
        buf
    };

    let parsed: serde_json::Value = serde_json::from_str(&features_json).expect("Invalid JSON");
    let features: Vec<f32> = parsed["features"]
        .as_array()
        .expect("Expected 'features' array")
        .iter()
        .map(|v| v.as_f64().expect("Expected number") as f32)
        .collect();

    assert_eq!(
        features.len(),
        NUM_FEATURES,
        "Expected {} features, got {}",
        NUM_FEATURES,
        features.len()
    );

    let input = Variable::new(
        Tensor::from_vec(features, &[1, NUM_FEATURES]).unwrap(),
        false,
    );

    let (health_score, severity_class, recon_mse) = model.assess(&input);

    // Output as JSON
    let output = serde_json::json!({
        "health_score": health_score,
        "severity": Sentinel::severity_name(severity_class),
        "severity_class": severity_class,
        "reconstruction_mse": recon_mse,
        "anomaly_detected": recon_mse > 0.05,
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
