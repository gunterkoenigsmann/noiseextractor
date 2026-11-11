//! Small numeric helpers shared by the analysis code.

/// Pearson correlation coefficient of two equally long series.
///
/// Returns 0 when either series is constant (or shorter than two samples),
/// because "no variation" carries no evidence of a shared modulation.
pub fn pearson(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    if n < 2 {
        return 0.0;
    }
    let (a, b) = (&a[..n], &b[..n]);
    let mean_a = a.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let mean_b = b.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
    let (mut cov, mut var_a, mut var_b) = (0.0f64, 0.0f64, 0.0f64);
    for (&x, &y) in a.iter().zip(b) {
        let dx = x as f64 - mean_a;
        let dy = y as f64 - mean_b;
        cov += dx * dy;
        var_a += dx * dx;
        var_b += dy * dy;
    }
    if var_a <= f64::EPSILON || var_b <= f64::EPSILON {
        return 0.0;
    }
    (cov / (var_a.sqrt() * var_b.sqrt())) as f32
}

/// Value below which the fraction `q` (0..=1) of `values` lies.
///
/// Uses selection on a scratch copy, so the input is left untouched.
/// Returns 0 for an empty slice.
pub fn quantile(values: &[f32], q: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut scratch = values.to_vec();
    let idx = ((scratch.len() - 1) as f32 * q.clamp(0.0, 1.0)).round() as usize;
    let (_, v, _) = scratch.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *v
}

/// Hermite smoothstep: 0 below `edge0`, 1 above `edge1`, smooth in between.
///
/// Used to turn a correlation score into a soft membership weight instead of
/// a hard threshold.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    if edge1 <= edge0 {
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Exponential integral E1(x) for x > 0.
///
/// Needed by the MMSE log-spectral-amplitude gain. Uses the power series for
/// small arguments and a continued fraction (modified Lentz) for large ones,
/// following Numerical Recipes §6.3. Accurate to ~1e-7 relative.
pub fn expint_e1(x: f64) -> f64 {
    const EULER: f64 = 0.577_215_664_901_532_9;
    if x <= 0.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        // E1(x) = -γ - ln x - Σ_{k≥1} (-x)^k / (k·k!)
        let mut sum = 0.0;
        let mut term = 1.0;
        for k in 1..60 {
            term *= -x / k as f64;
            let add = term / k as f64;
            sum += add;
            if add.abs() < 1e-16 * sum.abs().max(1e-300) {
                break;
            }
        }
        -EULER - x.ln() - sum
    } else {
        // Continued fraction for e^x·E1(x).
        let tiny = 1e-300;
        let mut b = x + 1.0;
        let mut c = 1.0 / tiny;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..200 {
            let an = -((i * i) as f64);
            b += 2.0;
            d = 1.0 / (an * d + b);
            c = b + an / c;
            let del = c * d;
            h *= del;
            if (del - 1.0).abs() < 1e-15 {
                break;
            }
        }
        h * (-x).exp()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pearson_detects_linear_relation_and_constant_series() {
        let a = [1.0, 2.0, 3.0, 4.0];
        assert!((pearson(&a, &[2.0, 4.0, 6.0, 8.0]) - 1.0).abs() < 1e-6);
        assert!((pearson(&a, &[8.0, 6.0, 4.0, 2.0]) + 1.0).abs() < 1e-6);
        assert_eq!(pearson(&a, &[5.0; 4]), 0.0);
        assert_eq!(pearson(&[1.0], &[1.0]), 0.0);
    }

    #[test]
    fn quantile_picks_order_statistics() {
        let v: Vec<f32> = (0..101).map(|x| x as f32).collect();
        assert_eq!(quantile(&v, 0.0), 0.0);
        assert_eq!(quantile(&v, 0.5), 50.0);
        assert_eq!(quantile(&v, 1.0), 100.0);
        assert_eq!(quantile(&[], 0.5), 0.0);
    }

    #[test]
    fn smoothstep_is_clamped_and_monotonic() {
        assert_eq!(smoothstep(0.2, 0.6, 0.0), 0.0);
        assert_eq!(smoothstep(0.2, 0.6, 1.0), 1.0);
        assert!((smoothstep(0.2, 0.6, 0.4) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn expint_matches_reference_values() {
        // Reference values from Abramowitz & Stegun table 5.1.
        for (x, e) in [(0.1, 1.822_923_958), (0.5, 0.559_773_595), (1.0, 0.219_383_934), (5.0, 0.001_148_296)] {
            let got = expint_e1(x);
            assert!(((got - e) / e).abs() < 1e-6, "E1({x}) = {got}, want {e}");
        }
    }
}
