//! NexusSentinel — Facility Health Assessment for Edge Deployment
//!
//! MLP autoencoder + health scorer + severity classifier that takes a
//! 48-dimensional facility feature vector and outputs:
//! - Reconstruction (facility anomaly via reconstruction error)
//! - Health score (0-1, higher = healthier)
//! - Severity classification (normal/watch/warning/critical)
//!
//! @version 0.2.0

pub mod config;
pub mod datagen;
pub mod sentinel;

pub use sentinel::Sentinel;
