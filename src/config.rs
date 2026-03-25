//! NexusSentinel — Feature Vector Configuration
//!
//! 48-dimensional facility health feature vector aggregated from all
//! equipment inference signals at a location. Extended from the original
//! 32-dim vector with temporal features and equipment-type breakdowns.

// =============================================================================
// Dimensions
// =============================================================================

/// Total features in the facility health input vector.
pub const NUM_FEATURES: usize = 48;

/// Latent dimension of the autoencoder bottleneck.
pub const LATENT_DIM: usize = 24;

/// Hidden dimension for encoder/decoder.
pub const HIDDEN_DIM: usize = 96;

/// Health head hidden dimension.
pub const HEALTH_HIDDEN: usize = 16;

/// Number of health severity classes for the classifier head.
pub const NUM_SEVERITY: usize = 4; // normal, watch, warning, critical

// =============================================================================
// Feature Vector Definition (48 features)
// =============================================================================

/// Ordered feature names matching the input tensor columns.
pub const FEATURE_NAMES: [&str; NUM_FEATURES] = [
    // --- Equipment Overview (0-3) ---
    "num_equipment",            // [0]  total equipment count (normalized)
    "pct_equipment_online",     // [1]  % of equipment with fresh data
    "equipment_diversity",      // [2]  count of distinct equipment types (normalized)
    "data_freshness",           // [3]  age of most recent data point (0=fresh, 1=stale)

    // --- Anomaly Detection Signals (4-9) ---
    "mean_anomaly_score",       // [4]  avg anomaly score across all equipment
    "max_anomaly_score",        // [5]  worst anomaly score
    "std_anomaly_score",        // [6]  variance in anomaly scores
    "pct_anomaly_flagged",      // [7]  % of equipment with anomaly_level != "normal"
    "anomaly_trend_slope",      // [8]  slope of anomaly scores over last hour
    "anomaly_acceleration",     // [9]  second derivative — is degradation accelerating?

    // --- FDD Predictions (10-17) ---
    "mean_fdd_confidence_5min", // [10] avg confidence of top FDD prediction (5min)
    "mean_fdd_confidence_15min",// [11] avg confidence (15min)
    "mean_fdd_confidence_30min",// [12] avg confidence (30min)
    "pct_fdd_failure_5min",     // [13] % equipment with is_failure=1 at 5min
    "pct_fdd_failure_15min",    // [14] % at 15min
    "pct_fdd_failure_30min",    // [15] % at 30min
    "num_distinct_faults",      // [16] diversity of predicted faults (normalized)
    "fdd_trend_slope",          // [17] is failure confidence increasing?

    // --- Alarm Status (18-21) ---
    "num_active_alarms",        // [18] current alarm count (normalized)
    "num_critical_alarms",      // [19] critical alarm count (normalized)
    "alarm_rate",               // [20] alarms per hour over lookback (normalized)
    "alarm_escalation_rate",    // [21] rate of alarm severity increase

    // --- Temperature Signals (22-31) ---
    "mean_supply_temp",         // [22] avg supply temp (normalized 0-1)
    "std_supply_temp",          // [23] variance in supply temps
    "mean_return_temp",         // [24] avg return temp
    "std_return_temp",          // [25] variance
    "mean_space_temp",          // [26] avg space temp
    "std_space_temp",           // [27] variance
    "temp_delta_mean",          // [28] avg (supply - return) delta (normalized)
    "temp_delta_std",           // [29] variance of deltas
    "pct_space_temp_oor",       // [30] % of zones with space temp out of setpoint range
    "max_space_temp_deviation", // [31] worst deviation from setpoint

    // --- Valve & Actuator Positions (32-35) ---
    "mean_hw_valve",            // [32] avg HW valve position (0-1)
    "mean_cw_valve",            // [33] avg CW valve position (0-1)
    "pct_valves_saturated",     // [34] % of valves at 0% or 100% (stuck/saturated)
    "mean_damper_position",     // [35] avg OA damper position

    // --- Fan & Motor Status (36-38) ---
    "pct_fans_running",         // [36] % of fans currently enabled
    "pct_fans_commanded_off",   // [37] % of fans that should be on but aren't
    "mean_vfd_speed",           // [38] avg VFD speed where applicable

    // --- Environmental (39-41) ---
    "outdoor_temp",             // [39] OAT (normalized)
    "outdoor_humidity",         // [40] humidity (normalized 0-1)
    "hvac_mode",                // [41] current mode (0=off, 0.25=eco, 0.5=heat, 0.75=cool, 1=emergency)

    // --- Temporal Patterns (42-44) ---
    "hour_of_day",              // [42] normalized 0-1 (0=midnight, 0.5=noon)
    "day_of_week",              // [43] normalized 0-1 (0=monday, 1=sunday)
    "is_occupied",              // [44] occupancy schedule (0 or 1)

    // --- Energy & Efficiency (45-47) ---
    "estimated_energy_usage",   // [45] normalized energy consumption
    "efficiency_score",         // [46] COP or efficiency metric (normalized)
    "energy_trend_slope",       // [47] is energy usage trending up?
];

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_count() {
        assert_eq!(FEATURE_NAMES.len(), NUM_FEATURES);
    }
}
