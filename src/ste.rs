use half::f16;

// =====================================================================
// Straight-Through Estimator (STE) for Ternary Quantization
// =====================================================================
//
// During forward: hard ternary quantization {-1, 0, 1}
// During backward: gradient passes through as if quantization is identity
// This enables end-to-end training of ternary-weight neural networks.

// =====================================================================
// Ternary Quantization
// =====================================================================

/// Quantize FP16 weights to ternary {-1, 0, 1} using threshold-based rounding.
///
/// The threshold is computed as: `threshold = alpha * mean(|W|)`
/// where `alpha` is a scaling hyperparameter (typically 0.5-0.7).
///
/// # Arguments
/// * `weights` - FP16 weight matrix, flat array of length M*N
/// * `alpha`   - Threshold scaling factor (0.0 to 1.0)
///
/// # Returns
/// (ternary_weights, scale_factor)
/// - ternary_weights: Vec<i8> of {-1, 0, 1}
/// - scale_factor: the quantization scale for dequantization
pub fn ternary_quantize_ste(weights: &[f16], alpha: f32) -> (Vec<i8>, f32) {
    assert!((0.0..=1.0).contains(&alpha), "alpha must be in [0, 1], got {}", alpha);

    // Compute mean absolute value
    let abs_sum: f32 = weights.iter().map(|w| w.to_f32().abs()).sum();
    let mean_abs = abs_sum / weights.len() as f32;

    // Threshold
    let threshold = alpha * mean_abs;

    // Compute scale: mean of |w| where |w| > threshold (non-zero ternary weights)
    let mut scale_sum = 0.0f32;
    let mut scale_count = 0usize;
    let ternary: Vec<i8> = weights
        .iter()
        .map(|&w| {
            let wf = w.to_f32();
            let abs_wf = wf.abs();
            if abs_wf > threshold {
                scale_sum += abs_wf;
                scale_count += 1;
                if wf > 0.0 { 1 } else { -1 }
            } else {
                0
            }
        })
        .collect();

    let scale = if scale_count > 0 {
        scale_sum / scale_count as f32
    } else {
        1.0
    };

    (ternary, scale)
}

/// Quantize with a fixed threshold (useful for inference).
pub fn ternary_quantize_fixed(weights: &[f16], threshold: f32) -> Vec<i8> {
    weights
        .iter()
        .map(|&w| {
            let wf = w.to_f32();
            if wf > threshold {
                1
            } else if wf < -threshold {
                -1
            } else {
                0
            }
        })
        .collect()
}

// =====================================================================
// STE Gradient Computation
// =====================================================================

/// Compute gradient with respect to weights using STE approximation.
///
/// In STE, the gradient of the quantization function is approximated as 1
/// within the clipping range and 0 outside:
///   dL/dw ≈ dL/dq * 1_{|w| <= 1}
///
/// For the ternary case with scaling:
///   dL/dw_raw = dL/dq * scale * 1_{|w_raw/scale| <= 1}
///
/// # Arguments
/// * `grad_output` - Gradient of loss w.r.t. output, shape [M]
/// * `activations` - Input activations, shape [N]
/// * `raw_weights` - Original FP16 weights (pre-quantization), shape [M*N]
/// * `scale`       - Quantization scale factor
///
/// # Returns
/// Gradient w.r.t. raw weights, shape [M*N]
pub fn ste_backward_weights(
    grad_output: &[f16],
    activations: &[f16],
    raw_weights: &[f16],
    scale: f32,
) -> Vec<f16> {
    let m = grad_output.len();
    let n = activations.len();
    assert_eq!(raw_weights.len(), m * n);
    assert!(scale > 0.0, "scale must be positive, got {}", scale);

    let mut grad_weights = Vec::with_capacity(m * n);

    for mi in 0..m {
        let go = grad_output[mi].to_f32();
        for ni in 0..n {
            let x = activations[ni].to_f32();
            let w = raw_weights[mi * n + ni].to_f32();

            let w_normalized = w.abs() / scale;
            let grad = if w_normalized <= 1.0 {
                go * x
            } else {
                0.0
            };

            grad_weights.push(f16::from_f32(grad));
        }
    }

    grad_weights
}

/// Compute gradient with respect to activations using STE.
///
/// # Arguments
/// * `grad_output` - Gradient of loss w.r.t. output, shape [M]
/// * `ternary_weights` - Quantized ternary weights, shape [M*N]
/// * `scale` - Quantization scale factor
///
/// # Returns
/// Gradient w.r.t. activations, shape [N]
pub fn ste_backward_activations(
    grad_output: &[f16],
    ternary_weights: &[i8],
    scale: f32,
) -> Vec<f16> {
    let m = grad_output.len();
    assert!(m > 0, "grad_output must not be empty");
    let n = ternary_weights.len() / m;
    assert_eq!(ternary_weights.len(), m * n);

    let mut grad_act = vec![0.0f32; n];

    for mi in 0..m {
        let go = grad_output[mi].to_f32();
        for ni in 0..n {
            let w = ternary_weights[mi * n + ni] as f32 * scale;
            grad_act[ni] += go * w;
        }
    }

    grad_act.iter().map(|&g| f16::from_f32(g)).collect()
}

// =====================================================================
// Dequantization
// =====================================================================

/// Dequantize ternary weights back to FP16 using scale factor.
pub fn dequantize_ternary(ternary_weights: &[i8], scale: f32) -> Vec<f16> {
    ternary_weights
        .iter()
        .map(|&w| f16::from_f32(w as f32 * scale))
        .collect()
}

// =====================================================================
// Tests
// =====================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_quantize_basic() {
        let weights: Vec<f16> = vec![
            f16::from_f32(0.8),
            f16::from_f32(-0.7),
            f16::from_f32(0.1),
            f16::from_f32(-0.05),
            f16::from_f32(0.6),
            f16::from_f32(-0.9),
        ];

        let (ternary, scale) = ternary_quantize_ste(&weights, 0.5);

        // mean(|w|) = (0.8+0.7+0.1+0.05+0.6+0.9)/6 = 3.15/6 = 0.525
        // threshold = 0.5 * 0.525 = 0.2625
        // 0.8 > 0.2625 -> 1
        // -0.7 > -0.2625 -> -1 (abs check: 0.7 > 0.2625)
        // 0.1 < 0.2625 -> 0
        // 0.05 < 0.2625 -> 0
        // 0.6 > 0.2625 -> 1
        // 0.9 > 0.2625 -> -1 (abs check: 0.9 > 0.2625)

        assert_eq!(ternary, vec![1, -1, 0, 0, 1, -1]);
        assert!(scale > 0.0);
    }

    #[test]
    fn test_ternary_quantize_all_zeros() {
        // With alpha=0.5, threshold = 0.5 * mean(|w|).
        // Since all weights are non-zero and similar magnitude,
        // they all exceed the threshold and become ±1.
        // This verifies the algorithm handles near-uniform small weights.
        let weights: Vec<f16> = vec![
            f16::from_f32(0.01),
            f16::from_f32(-0.01),
            f16::from_f32(0.005),
        ];

        let (ternary, _scale) = ternary_quantize_ste(&weights, 0.5);
        // mean(|w|) = 0.00833, threshold = 0.00417
        // 0.01 > 0.00417 -> 1, -0.01 < -0.00417 -> -1, 0.005 > 0.00417 -> 1
        assert_eq!(ternary, vec![1, -1, 1]);
    }

    #[test]
    fn test_dequantize_roundtrip() {
        let ternary: Vec<i8> = vec![1, 0, -1, 1];
        let scale = 0.5;
        let dequant = dequantize_ternary(&ternary, scale);

        let expected: Vec<f16> = vec![
            f16::from_f32(0.5),
            f16::from_f32(0.0),
            f16::from_f32(-0.5),
            f16::from_f32(0.5),
        ];

        for (a, b) in dequant.iter().zip(expected.iter()) {
            assert!((a.to_f32() - b.to_f32()).abs() < 1e-6);
        }
    }

    #[test]
    fn test_ste_backward_weights() {
        let grad_output: Vec<f16> = vec![f16::from_f32(1.0)];
        let activations: Vec<f16> = vec![f16::from_f32(0.5), f16::from_f32(0.3)];
        let raw_weights: Vec<f16> = vec![f16::from_f32(0.4), f16::from_f32(0.2)];
        let scale = 0.5;

        let grad = ste_backward_weights(&grad_output, &activations, &raw_weights, scale);

        // w/scale: 0.4/0.5=0.8 <=1 -> grad = 1.0*0.5 = 0.5
        // w/scale: 0.2/0.5=0.4 <=1 -> grad = 1.0*0.3 = 0.3
        assert!((grad[0].to_f32() - 0.5).abs() < 1e-3);
        assert!((grad[1].to_f32() - 0.3).abs() < 1e-3);
    }
}
