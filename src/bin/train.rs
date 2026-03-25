//! NexusSentinel Training
//!
//! Trains the facility health model with:
//! - Reconstruction MSE loss (autoencoder)
//! - Health score MSE loss (regression)
//! - Severity cross-entropy loss (classification)
//! - Live training monitor
//! - Checkpoint save/resume
//!
//! Usage: cargo run --release --bin train

use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use axonml_autograd::Variable;
use axonml_nn::{CrossEntropyLoss, Module};
use axonml_optim::{
    lr_scheduler::{CosineAnnealingLR, LRScheduler},
    Adam, Optimizer,
};
use axonml_serialize::{
    load_checkpoint, save_checkpoint, save_model, Checkpoint, StateDict, TrainingState,
};
use axonml_tensor::Tensor;

use nexus_sentinel::config::{NUM_FEATURES, NUM_SEVERITY};
use nexus_sentinel::datagen::{generate_batch, Rng};
use nexus_sentinel::Sentinel;

fn main() {
    println!("═══════════════════════════════════════════════════════════");
    println!(" NexusSentinel v0.2 — Facility Health Assessment");
    println!(" Architecture: MLP Autoencoder + Health + Severity");
    println!("═══════════════════════════════════════════════════════════");
    println!();

    let output_dir = PathBuf::from("/opt/NexusSentinel/checkpoints");
    fs::create_dir_all(&output_dir).ok();

    let mut model = Sentinel::new();
    let total_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!("  Parameters: {}", total_params);
    println!("  Input: [B, {}]", NUM_FEATURES);
    println!("  Severity classes: {}", NUM_SEVERITY);

    // Hyperparameters
    let num_epochs = 200;
    let batch_size = 64;
    let batches_per_epoch = 32;
    let learning_rate = 0.002;
    let recon_weight = 1.0;
    let health_weight = 3.0;
    let severity_weight = 1.5;

    // Launch monitor
    let monitor = axonml::monitor::TrainingMonitor::new("NexusSentinel", total_params)
        .total_epochs(num_epochs)
        .batch_size(batch_size * batches_per_epoch)
        .launch();

    let params = model.parameters();
    let mut optimizer = Adam::new(params, learning_rate);
    let mut scheduler = CosineAnnealingLR::with_eta_min(&optimizer, num_epochs, 1e-6);
    let ce_loss = CrossEntropyLoss::new();

    // Resume from checkpoint
    let mut training_state = TrainingState::new();
    let mut best_loss = f32::INFINITY;
    let mut start_epoch = 0;

    let latest_ckpt = output_dir.join("checkpoint_latest.axonml");
    if latest_ckpt.exists() {
        if let Ok(checkpoint) = load_checkpoint(&latest_ckpt) {
            let model_params = model.named_parameters();
            let mut loaded = 0;
            for (name, param) in &model_params {
                if let Some(entry) = checkpoint.model_state.get(name) {
                    if let Ok(tensor) = entry.data.to_tensor() {
                        if tensor.shape() == param.data().shape() {
                            param.update_data(tensor);
                            loaded += 1;
                        }
                    }
                }
            }
            start_epoch = checkpoint.epoch();
            if let Some(bl) = checkpoint.training_state.best_metric {
                best_loss = bl;
            }
            training_state = checkpoint.training_state;
            println!("  Resumed from epoch {} ({}/{} params, best: {:.6})",
                start_epoch, loaded, model_params.len(), best_loss);
        }
    }

    let mut rng = Rng::new(42 + start_epoch as u64 * 1000);

    println!();
    println!("  Epochs: {}-{}, Batch: {}, Batches/epoch: {}",
        start_epoch + 1, num_epochs, batch_size, batches_per_epoch);
    println!("  Weights: recon={}, health={}, severity={}",
        recon_weight, health_weight, severity_weight);
    println!("  LR: {} → CosineAnnealing", learning_rate);
    println!();

    let training_start = Instant::now();

    for epoch in start_epoch..num_epochs {
        model.set_training(true);
        let mut epoch_recon = 0.0f32;
        let mut epoch_health = 0.0f32;
        let mut epoch_severity = 0.0f32;
        let mut epoch_total = 0.0f32;

        for _ in 0..batches_per_epoch {
            let (feat_data, label_data, sev_data) = generate_batch(&mut rng, batch_size);

            let input = Variable::new(
                Tensor::from_vec(feat_data.clone(), &[batch_size, NUM_FEATURES]).unwrap(),
                true,
            );
            let target = Variable::new(
                Tensor::from_vec(feat_data, &[batch_size, NUM_FEATURES]).unwrap(),
                false,
            );
            let health_labels = Variable::new(
                Tensor::from_vec(label_data, &[batch_size, 1]).unwrap(),
                false,
            );
            // Severity labels as class indices
            let sev_labels = Variable::new(
                Tensor::from_vec(
                    sev_data.iter().map(|&s| s as f32).collect(),
                    &[batch_size],
                )
                .unwrap(),
                false,
            );

            let (_latent, reconstructed, health_score, severity_logits) =
                model.forward_all(&input);

            // Reconstruction MSE
            let recon_diff = reconstructed.sub_var(&target);
            let recon_loss = recon_diff.mul_var(&recon_diff).mean();

            // Health MSE
            let health_diff = health_score.sub_var(&health_labels);
            let health_loss = health_diff.mul_var(&health_diff).mean();

            // Severity cross-entropy
            let severity_loss = ce_loss.compute(&severity_logits, &sev_labels);

            // Combined loss
            let total_loss = recon_loss
                .mul_scalar(recon_weight)
                .add_var(&health_loss.mul_scalar(health_weight))
                .add_var(&severity_loss.mul_scalar(severity_weight));

            total_loss.backward();
            optimizer.step();
            optimizer.zero_grad();

            epoch_recon += recon_loss.data().to_vec()[0];
            epoch_health += health_loss.data().to_vec()[0];
            epoch_severity += severity_loss.data().to_vec()[0];
            epoch_total += total_loss.data().to_vec()[0];

            training_state.next_step();
        }

        let n = batches_per_epoch as f32;
        let avg_recon = epoch_recon / n;
        let avg_health = epoch_health / n;
        let avg_severity = epoch_severity / n;
        let avg_total = epoch_total / n;

        scheduler.step(&mut optimizer);
        let lr = scheduler.get_last_lr();

        training_state.record_loss(avg_total);

        // Log to monitor
        monitor.log_epoch(
            epoch + 1,
            avg_total,
            None,
            vec![
                ("recon", avg_recon),
                ("health", avg_health),
                ("severity", avg_severity),
                ("lr", lr),
            ],
        );

        let improved = if avg_total < best_loss {
            best_loss = avg_total;
            training_state.update_best("loss", avg_total, false);

            save_model(&model, output_dir.join("best_model.axonml")).ok();
            let best_ckpt = Checkpoint::builder()
                .model_state(StateDict::from_module(&model))
                .training_state(training_state.clone())
                .epoch(epoch + 1)
                .build();
            save_checkpoint(&best_ckpt, output_dir.join("checkpoint_best.axonml")).ok();
            " *"
        } else {
            ""
        };

        // Always save latest
        let latest = Checkpoint::builder()
            .model_state(StateDict::from_module(&model))
            .training_state(training_state.clone())
            .epoch(epoch + 1)
            .build();
        save_checkpoint(&latest, output_dir.join("checkpoint_latest.axonml")).ok();

        if (epoch + 1) % 20 == 0 {
            save_checkpoint(
                &latest,
                output_dir.join(format!("checkpoint_epoch_{:04}.axonml", epoch + 1)),
            )
            .ok();
        }

        if (epoch + 1) % 10 == 0 || epoch == start_epoch || epoch == num_epochs - 1 {
            println!(
                "Epoch {:3}/{} | total: {:.6} | recon: {:.6} | health: {:.6} | sev: {:.6} | lr: {:.8}{}",
                epoch + 1, num_epochs, avg_total, avg_recon, avg_health, avg_severity, lr, improved
            );
        }

        training_state.next_epoch();
    }

    // Validation
    println!();
    println!("--- Validation ---");
    model.set_training(false);

    use nexus_sentinel::datagen::{generate_sample, Condition};
    for (name, condition) in [
        ("Normal", Condition::Normal),
        ("Watch", Condition::Watch),
        ("Warning", Condition::Warning),
        ("Critical", Condition::Critical),
    ] {
        let (features, expected, expected_sev) = generate_sample(&mut rng, condition);
        let input = Variable::new(
            Tensor::from_vec(features, &[1, NUM_FEATURES]).unwrap(),
            false,
        );
        let (score, sev_class, mse) = model.assess(&input);
        println!(
            "  {:10} | health: {:.4} (exp: {:.2}) | severity: {} (exp: {}) | recon_mse: {:.6}",
            name, score, expected, Sentinel::severity_name(sev_class),
            Sentinel::severity_name(expected_sev), mse
        );
    }

    // Save final model + export
    save_model(&model, PathBuf::from("/opt/NexusSentinel/models/sentinel.axonml"))
        .expect("Failed to save model");

    monitor.set_status("complete");

    let total_time = training_start.elapsed();
    println!();
    println!("═══════════════════════════════════════════════════════════");
    println!(
        " Done — {:.1}s ({:.1} min) | Best: {:.6}",
        total_time.as_secs_f32(),
        total_time.as_secs_f32() / 60.0,
        best_loss
    );
    println!(" Model: /opt/NexusSentinel/models/sentinel.axonml");
    println!("═══════════════════════════════════════════════════════════");
}
