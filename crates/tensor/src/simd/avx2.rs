use std::arch::x86_64::*;

#[cfg(target_arch = "x86_64")]
pub fn detect_simd_info() -> &'static str {
    if is_x86_feature_detected!("avx512f") {
        "avx512"
    } else if is_x86_feature_detected!("avx2") {
        "avx2"
    } else if is_x86_feature_detected!("sse4.1") {
        "sse4.1"
    } else {
        "sse2"
    }
}

/// # Safety
/// `a` and `b` must be valid for reads and `out` for writes of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_add_avx2(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vb = _mm256_loadu_ps(b.add(base));
        let vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = *a.add(offset + i) + *b.add(offset + i);
    }
}

/// # Safety
/// `a` and `b` must be valid for reads and `out` for writes of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_sub_avx2(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vb = _mm256_loadu_ps(b.add(base));
        let vr = _mm256_sub_ps(va, vb);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = *a.add(offset + i) - *b.add(offset + i);
    }
}

/// # Safety
/// `a` and `b` must be valid for reads and `out` for writes of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_mul_avx2(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vb = _mm256_loadu_ps(b.add(base));
        let vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = *a.add(offset + i) * *b.add(offset + i);
    }
}

/// # Safety
/// `a` must be valid for reads and `out` for writes of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_scale_avx2(a: *const f32, scale: f32, out: *mut f32, len: usize) {
    let v_scale = _mm256_set1_ps(scale);
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vr = _mm256_mul_ps(va, v_scale);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = *a.add(offset + i) * scale;
    }
}

/// # Safety
/// `a` and `b` must be valid for reads and `out` for writes of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_mul_scaled_avx2(
    a: *const f32,
    b: *const f32,
    scale: f32,
    out: *mut f32,
    len: usize,
) {
    let v_scale = _mm256_set1_ps(scale);
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vb = _mm256_loadu_ps(b.add(base));
        let vr = _mm256_mul_ps(va, vb);
        let vr = _mm256_mul_ps(vr, v_scale);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = *a.add(offset + i) * *b.add(offset + i) * scale;
    }
}

/// # Safety
/// `a` must be valid for reads and `out` for writes of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_add_scalar_avx2(a: *const f32, scalar: f32, out: *mut f32, len: usize) {
    let v_scalar = _mm256_set1_ps(scalar);
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vr = _mm256_add_ps(va, v_scalar);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = *a.add(offset + i) + scalar;
    }
}

/// # Safety
/// `a` must be valid for reads and `b` for reads and writes of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_axpy_avx2(a: *const f32, scale: f32, b: *mut f32, len: usize) {
    let v_scale = _mm256_set1_ps(scale);
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vb = _mm256_loadu_ps(b.add(base));
        let vr = _mm256_fmadd_ps(va, v_scale, vb);
        _mm256_storeu_ps(b.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *b.add(offset + i) += *a.add(offset + i) * scale;
    }
}

/// # Safety
/// `a` and `b` must be valid for reads of `len` f32 elements.
#[target_feature(enable = "avx2,fma")]
pub unsafe fn f32_dot_avx2(a: *const f32, b: *const f32, len: usize) -> f32 {
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vb = _mm256_loadu_ps(b.add(base));
        acc = _mm256_fmadd_ps(va, vb, acc);
    }

    let mut result = hsum256(acc);

    let offset = chunks * 8;
    for i in 0..remainder {
        result += *a.add(offset + i) * *b.add(offset + i);
    }

    result
}

#[target_feature(enable = "avx2,fma")]
unsafe fn hsum256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(hi, lo);

    let hi64 = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, hi64);

    let hi32 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, hi32);

    _mm_cvtss_f32(sum32)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn f32_exp_avx2(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / 8;
    let remainder = len % 8;

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vr = fast_exp256(va);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = (*a.add(offset + i)).exp();
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn f32_silu_avx2(a: *const f32, out: *mut f32, len: usize) {
    let chunks = len / 8;
    let remainder = len % 8;
    let one = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0);

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let neg = _mm256_xor_ps(va, sign_mask);
        let exp_neg = fast_exp256(neg);
        let denom = _mm256_add_ps(one, exp_neg);
        let vr = _mm256_div_ps(va, denom);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        *out.add(offset + i) = *a.add(offset + i) / (1.0 + (-*a.add(offset + i)).exp());
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn f32_silu_mul_avx2(a: *const f32, b: *const f32, out: *mut f32, len: usize) {
    let chunks = len / 8;
    let remainder = len % 8;
    let one = _mm256_set1_ps(1.0);
    let sign_mask = _mm256_set1_ps(-0.0);

    for i in 0..chunks {
        let base = i * 8;
        let va = _mm256_loadu_ps(a.add(base));
        let vb = _mm256_loadu_ps(b.add(base));
        let neg = _mm256_xor_ps(va, sign_mask);
        let exp_neg = fast_exp256(neg);
        let denom = _mm256_add_ps(one, exp_neg);
        let silu = _mm256_div_ps(va, denom);
        let vr = _mm256_mul_ps(silu, vb);
        _mm256_storeu_ps(out.add(base), vr);
    }

    let offset = chunks * 8;
    for i in 0..remainder {
        let silu_i = *a.add(offset + i) / (1.0 + (-*a.add(offset + i)).exp());
        *out.add(offset + i) = silu_i * *b.add(offset + i);
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn fast_exp256(x: __m256) -> __m256 {
    #[allow(clippy::approx_constant, clippy::excessive_precision)]
    let ln2_inv = _mm256_set1_ps(1.442_695_040_9f32);
    let ln2_hi = _mm256_set1_ps(0.693_359_4f32);
    let ln2_lo = _mm256_set1_ps(-2.121_944_4e-4);
    let one = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(1.0 / 2.0);
    let c2 = _mm256_set1_ps(1.0 / 6.0);
    let c3 = _mm256_set1_ps(1.0 / 24.0);
    let c4 = _mm256_set1_ps(1.0 / 120.0);
    let c5 = _mm256_set1_ps(1.0 / 720.0);

    // Clamp to the f32 representable range. Beyond ~±87.3 the true exp either
    // overflows to inf or underflows to 0; without the clamp, the integer
    // exponent in `fast_pow2_256` leaves the biased range and produces
    // garbage (NaN / huge wrong signs) instead of 0/inf.
    let max_x = _mm256_set1_ps(88.72);
    let min_x = _mm256_set1_ps(-87.34);
    let x = _mm256_min_ps(_mm256_max_ps(x, min_x), max_x);

    let x_ln2 = _mm256_mul_ps(x, ln2_inv);
    let sign_mask = _mm256_set1_ps(-0.0);
    let half = _mm256_set1_ps(0.5);
    let adjust = _mm256_or_ps(_mm256_and_ps(x_ln2, sign_mask), half);
    let n_f = _mm256_cvtepi32_ps(_mm256_cvtps_epi32(_mm256_add_ps(x_ln2, adjust)));

    let r = _mm256_sub_ps(
        _mm256_sub_ps(x, _mm256_mul_ps(n_f, ln2_hi)),
        _mm256_mul_ps(n_f, ln2_lo),
    );

    // Horner: exp(r) ≈ 1 + r + r²/2! + r³/3! + r⁴/4! + r⁵/5! + r⁶/6!
    let mut poly = c5;
    poly = _mm256_fmadd_ps(poly, r, c4);
    poly = _mm256_fmadd_ps(poly, r, c3);
    poly = _mm256_fmadd_ps(poly, r, c2);
    poly = _mm256_fmadd_ps(poly, r, c1);
    poly = _mm256_fmadd_ps(poly, r, one);
    poly = _mm256_fmadd_ps(poly, r, one);

    let pow2 = fast_pow2_256(_mm256_cvtps_epi32(n_f));
    _mm256_mul_ps(pow2, poly)
}

#[target_feature(enable = "avx2,fma")]
unsafe fn fast_pow2_256(xi: __m256i) -> __m256 {
    let bias = _mm256_set1_epi32(127);
    let shifted = _mm256_slli_epi32(_mm256_add_epi32(xi, bias), 23);
    _mm256_castsi256_ps(shifted)
}

pub fn f32_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_add_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    }
}

pub fn f32_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_sub_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i] - b[i];
        }
    }
}

pub fn f32_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_mul_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i] * b[i];
        }
    }
}

pub fn f32_scale(a: &[f32], scale: f32, out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_scale_avx2(a.as_ptr(), scale, out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i] * scale;
        }
    }
}

pub fn f32_mul_scaled(a: &[f32], b: &[f32], scale: f32, out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_mul_scaled_avx2(a.as_ptr(), b.as_ptr(), scale, out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i] * b[i] * scale;
        }
    }
}

pub fn f32_add_scalar(a: &[f32], scalar: f32, out: &mut [f32]) {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_add_scalar_avx2(a.as_ptr(), scalar, out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i] + scalar;
        }
    }
}

pub fn f32_add_scalar_inplace(a: &mut [f32], scalar: f32) {
    let n = a.len();
    if is_x86_feature_detected!("avx2") && n >= 8 {
        unsafe { f32_add_scalar_avx2(a.as_ptr(), scalar, a.as_mut_ptr(), n) };
    } else {
        for v in a.iter_mut().take(n) {
            *v += scalar;
        }
    }
}

pub fn f32_axpy(a: &[f32], scale: f32, b: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_axpy_avx2(a.as_ptr(), scale, b.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            b[i] += a[i] * scale;
        }
    }
}

pub fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_dot_avx2(a.as_ptr(), b.as_ptr(), a.len()) }
    } else {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
}

pub fn f32_sum(a: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe {
            let ptr = a.as_ptr();
            let len = a.len();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut acc = _mm256_setzero_ps();
            for i in 0..chunks {
                let base = i * 8;
                let v = _mm256_loadu_ps(ptr.add(base));
                acc = _mm256_add_ps(acc, v);
            }

            let mut result = hsum256(acc);
            let offset = chunks * 8;
            for i in 0..remainder {
                result += *ptr.add(offset + i);
            }
            result
        }
    } else {
        a.iter().sum()
    }
}

pub fn f32_max(a: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe {
            let ptr = a.as_ptr();
            let len = a.len();
            let chunks = len / 8;
            let remainder = len % 8;

            let mut acc = _mm256_set1_ps(f32::NEG_INFINITY);
            for i in 0..chunks {
                let base = i * 8;
                let v = _mm256_loadu_ps(ptr.add(base));
                acc = _mm256_max_ps(acc, v);
            }

            let mut result = hmax256(acc);
            let offset = chunks * 8;
            for i in 0..remainder {
                result = result.max(*ptr.add(offset + i));
            }
            result
        }
    } else {
        a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn hmax256(v: __m256) -> f32 {
    let hi = _mm256_extractf128_ps(v, 1);
    let lo = _mm256_castps256_ps128(v);
    let max128 = _mm_max_ps(hi, lo);

    let hi64 = _mm_movehl_ps(max128, max128);
    let max64 = _mm_max_ps(max128, hi64);

    let hi32 = _mm_shuffle_ps(max64, max64, 1);
    let max32 = _mm_max_ss(max64, hi32);

    _mm_cvtss_f32(max32)
}

pub fn f32_exp(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_exp_avx2(a.as_ptr(), out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i].exp();
        }
    }
}

pub fn f32_exp_inplace(a: &mut [f32]) {
    let n = a.len();
    if is_x86_feature_detected!("avx2") && n >= 8 {
        unsafe { f32_exp_avx2(a.as_ptr(), a.as_mut_ptr(), n) };
    } else {
        for v in a.iter_mut().take(n) {
            *v = v.exp();
        }
    }
}

pub fn f32_scale_inplace(a: &mut [f32], scale: f32) {
    let n = a.len();
    if is_x86_feature_detected!("avx2") && n >= 8 {
        unsafe { f32_scale_avx2(a.as_ptr(), scale, a.as_mut_ptr(), n) };
    } else {
        for v in a.iter_mut().take(n) {
            *v *= scale;
        }
    }
}

pub fn f32_silu(a: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), out.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_silu_avx2(a.as_ptr(), out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = a[i] / (1.0 + (-a[i]).exp());
        }
    }
}

/// Exact GELU: `0.5*x*(1 + erf(x/sqrt(2)))`.
pub fn f32_gelu(a: &[f32], out: &mut [f32]) {
    const INV_SQRT_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    for i in 0..a.len() {
        let x = a[i];
        out[i] = 0.5 * x * (1.0 + erf(x * INV_SQRT_2));
    }
}

/// Tanh-approximated GELU (GPT-2 `gelu_new` / Gemma `gelu_pytorch_tanh`):
/// `0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3)))`.
pub fn f32_gelu_tanh(a: &[f32], out: &mut [f32]) {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    for i in 0..a.len() {
        let x = a[i];
        out[i] = 0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh());
    }
}

/// Scalar (non-SIMD) approximation of `erf` (Abramowitz & Stegun 7.1.26).
fn erf(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let t = 1.0 / (1.0 + 0.3275911 * x.abs());
    let y = 1.0
        - (((((1.061_405_4 * t - 1.453_152_1) * t + 1.421_413_8) * t - 0.284_496_72) * t
            + 0.254_829_6)
            * t)
            * (-x * x).exp();
    sign * y
}

pub fn f32_silu_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), out.len());
    if is_x86_feature_detected!("avx2") && a.len() >= 8 {
        unsafe { f32_silu_mul_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), a.len()) };
    } else {
        for i in 0..a.len() {
            out[i] = (a[i] / (1.0 + (-a[i]).exp())) * b[i];
        }
    }
}

#[target_feature(enable = "avx2")]
unsafe fn xnor_popcount_2bit_avx2(a: *const u8, b: *const u8, out: *mut u8, n: usize) {
    // For each 4-bit nibble, the low 2 bits are the first 2-bit element and the
    // high 2 bits are the second.  The LUTs produce a 4-bit mask: bit j = 1 iff
    // element j of the nibble is +1 (01) or -1 (10).
    const LUT_01_LO: [u8; 16] = [
        0x00, 0x01, 0x00, 0x00, 0x02, 0x03, 0x02, 0x02,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00,
    ];
    const LUT_01_HI: [u8; 16] = [
        0x00, 0x04, 0x00, 0x00, 0x08, 0x0C, 0x08, 0x08,
        0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00,
    ];
    const LUT_10_LO: [u8; 16] = [
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00,
        0x02, 0x02, 0x03, 0x02, 0x00, 0x00, 0x01, 0x00,
    ];
    const LUT_10_HI: [u8; 16] = [
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x04, 0x00,
        0x08, 0x08, 0x0C, 0x08, 0x00, 0x00, 0x04, 0x00,
    ];

    let lut_01_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(LUT_01_LO.as_ptr() as *const __m128i));
    let lut_01_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(LUT_01_HI.as_ptr() as *const __m128i));
    let lut_10_lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(LUT_10_LO.as_ptr() as *const __m128i));
    let lut_10_hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(LUT_10_HI.as_ptr() as *const __m128i));

    let low_nibble_mask = _mm256_set1_epi8(0x0F);
    // Multipliers for _mm256_maddubs_epi16: (1, 16) per byte pair.
    let mul_1_16 = _mm256_set1_epi16(0x1001);
    let pack_shuffle = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
        0, 2, 4, 6, 8, 10, 12, 14, -1, -1, -1, -1, -1, -1, -1, -1,
    );

    let chunks = n / 128;

    for i in 0..chunks {
        let byte_idx = i * 32;
        let va = _mm256_loadu_si256(a.add(byte_idx) as *const __m256i);
        let vb = _mm256_loadu_si256(b.add(byte_idx) as *const __m256i);

        let va_lo = _mm256_and_si256(va, low_nibble_mask);
        let va_hi = _mm256_and_si256(_mm256_srli_epi16(va, 4), low_nibble_mask);
        let m01_a = _mm256_or_si256(
            _mm256_shuffle_epi8(lut_01_lo, va_lo),
            _mm256_shuffle_epi8(lut_01_hi, va_hi),
        );
        let m10_a = _mm256_or_si256(
            _mm256_shuffle_epi8(lut_10_lo, va_lo),
            _mm256_shuffle_epi8(lut_10_hi, va_hi),
        );

        let vb_lo = _mm256_and_si256(vb, low_nibble_mask);
        let vb_hi = _mm256_and_si256(_mm256_srli_epi16(vb, 4), low_nibble_mask);
        let m01_b = _mm256_or_si256(
            _mm256_shuffle_epi8(lut_01_lo, vb_lo),
            _mm256_shuffle_epi8(lut_01_hi, vb_hi),
        );
        let m10_b = _mm256_or_si256(
            _mm256_shuffle_epi8(lut_10_lo, vb_lo),
            _mm256_shuffle_epi8(lut_10_hi, vb_hi),
        );

        let same_01 = _mm256_and_si256(m01_a, m01_b);
        let same_10 = _mm256_and_si256(m10_a, m10_b);
        let mask4 = _mm256_and_si256(_mm256_or_si256(same_01, same_10), low_nibble_mask);

        // Pack pairs of 4-bit masks into bytes.
        let packed16 = _mm256_maddubs_epi16(mask4, mul_1_16);
        let shuffled = _mm256_shuffle_epi8(packed16, pack_shuffle);
        let lo = _mm256_castsi256_si128(shuffled);
        let hi = _mm256_extracti128_si256::<1>(shuffled);

        // Only the low 8 bytes of each lane contain useful data.
        _mm_storel_epi64(out.add(i * 16) as *mut __m128i, lo);
        _mm_storel_epi64(out.add(i * 16 + 8) as *mut __m128i, hi);
    }

    let offset = chunks * 128;
    for i in offset..n {
        let byte_a = *a.add(i / 4);
        let byte_b = *b.add(i / 4);
        let shift = (i % 4) * 2;
        let bits_a = (byte_a >> shift) & 0x03;
        let bits_b = (byte_b >> shift) & 0x03;

        let w_a: f32 = match bits_a {
            0x01 => 1.0,
            0x02 => -1.0,
            _ => 0.0,
        };
        let w_b: f32 = match bits_b {
            0x01 => 1.0,
            0x02 => -1.0,
            _ => 0.0,
        };

        if i % 8 == 0 {
            *out.add(i / 8) = 0;
        }
        if w_a * w_b > 0.0 {
            *out.add(i / 8) |= 1 << (i % 8);
        }
    }
}

#[target_feature(enable = "avx2,fma")]
unsafe fn f32_matmul_row_avx2_unrolled(
    a: *const f32,
    b_t: *const f32,
    out: *mut f32,
    k: usize,
    n: usize,
) {
    let mut j = 0;
    while j + 4 <= n {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut t = 0;
        while t + 8 <= k {
            let va = _mm256_loadu_ps(a.add(t));
            s0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add(j * k + t)), s0);
            s1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add((j + 1) * k + t)), s1);
            s2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add((j + 2) * k + t)), s2);
            s3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add((j + 3) * k + t)), s3);
            t += 8;
        }
        let mut sum0 = hsum256(s0);
        let mut sum1 = hsum256(s1);
        let mut sum2 = hsum256(s2);
        let mut sum3 = hsum256(s3);
        while t < k {
            let av = *a.add(t);
            sum0 += av * *b_t.add(j * k + t);
            sum1 += av * *b_t.add((j + 1) * k + t);
            sum2 += av * *b_t.add((j + 2) * k + t);
            sum3 += av * *b_t.add((j + 3) * k + t);
            t += 1;
        }
        *out.add(j) = sum0;
        *out.add(j + 1) = sum1;
        *out.add(j + 2) = sum2;
        *out.add(j + 3) = sum3;
        j += 4;
    }
    while j < n {
        let mut s = _mm256_setzero_ps();
        let mut t = 0;
        while t + 8 <= k {
            let va = _mm256_loadu_ps(a.add(t));
            s = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add(j * k + t)), s);
            t += 8;
        }
        let mut sum = hsum256(s);
        while t < k {
            sum += *a.add(t) * *b_t.add(j * k + t);
            t += 1;
        }
        *out.add(j) = sum;
        j += 1;
    }
}

/// Same as `f32_matmul_row_avx2_unrolled` but with separate stride (`k_stride`)
/// and length (`kk_len`), and accumulates into `out` instead of overwriting.
#[target_feature(enable = "avx2,fma")]
unsafe fn f32_matmul_row_partial_avx2(
    a: *const f32,
    b_t: *const f32,
    out: *mut f32,
    k_stride: usize,
    kk_len: usize,
    n: usize,
) {
    let mut j = 0;
    while j + 4 <= n {
        let mut s0 = _mm256_setzero_ps();
        let mut s1 = _mm256_setzero_ps();
        let mut s2 = _mm256_setzero_ps();
        let mut s3 = _mm256_setzero_ps();
        let mut t = 0;
        while t + 8 <= kk_len {
            let va = _mm256_loadu_ps(a.add(t));
            s0 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add(j * k_stride + t)), s0);
            s1 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add((j + 1) * k_stride + t)), s1);
            s2 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add((j + 2) * k_stride + t)), s2);
            s3 = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add((j + 3) * k_stride + t)), s3);
            t += 8;
        }
        let mut sum0 = hsum256(s0);
        let mut sum1 = hsum256(s1);
        let mut sum2 = hsum256(s2);
        let mut sum3 = hsum256(s3);
        while t < kk_len {
            let av = *a.add(t);
            sum0 += av * *b_t.add(j * k_stride + t);
            sum1 += av * *b_t.add((j + 1) * k_stride + t);
            sum2 += av * *b_t.add((j + 2) * k_stride + t);
            sum3 += av * *b_t.add((j + 3) * k_stride + t);
            t += 1;
        }
        *out.add(j) += sum0;
        *out.add(j + 1) += sum1;
        *out.add(j + 2) += sum2;
        *out.add(j + 3) += sum3;
        j += 4;
    }
    while j < n {
        let mut s = _mm256_setzero_ps();
        let mut t = 0;
        while t + 8 <= kk_len {
            let va = _mm256_loadu_ps(a.add(t));
            s = _mm256_fmadd_ps(va, _mm256_loadu_ps(b_t.add(j * k_stride + t)), s);
            t += 8;
        }
        let mut sum = hsum256(s);
        while t < kk_len {
            sum += *a.add(t) * *b_t.add(j * k_stride + t);
            t += 1;
        }
        *out.add(j) += sum;
        j += 1;
    }
}

pub fn f32_matmul_row(a: &[f32], b_t: &[f32], out_row: &mut [f32], k: usize, n: usize) {
    if is_x86_feature_detected!("avx2") && k >= 8 {
        unsafe {
            f32_matmul_row_avx2_unrolled(a.as_ptr(), b_t.as_ptr(), out_row.as_mut_ptr(), k, n)
        };
    } else {
        for j in (0..n).step_by(4) {
            let remaining = n - j;
            let mut s0 = 0.0f32;
            let mut s1 = 0.0f32;
            let mut s2 = 0.0f32;
            let mut s3 = 0.0f32;
            for t in 0..k {
                let av = a[t];
                s0 += av * b_t[j * k + t];
                if remaining > 1 {
                    s1 += av * b_t[(j + 1) * k + t];
                }
                if remaining > 2 {
                    s2 += av * b_t[(j + 2) * k + t];
                }
                if remaining > 3 {
                    s3 += av * b_t[(j + 3) * k + t];
                }
            }
            out_row[j] = s0;
            if remaining > 1 {
                out_row[j + 1] = s1;
            }
            if remaining > 2 {
                out_row[j + 2] = s2;
            }
            if remaining > 3 {
                out_row[j + 3] = s3;
            }
        }
    }
}

const KB: usize = 128;

pub fn f32_matmul(a: &[f32], b_t: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    if k <= KB {
        for i in 0..m {
            f32_matmul_row(&a[i * k..], b_t, &mut out[i * n..], k, n);
        }
        return;
    }
    out.fill(0.0);
    let mut kk = 0;
    while kk < k {
        let kk_end = (kk + KB).min(k);
        let kk_len = kk_end - kk;
        if is_x86_feature_detected!("avx2") && kk_len >= 8 {
            for i in 0..m {
                unsafe {
                    f32_matmul_row_partial_avx2(
                        a.as_ptr().add(i * k + kk),
                        b_t.as_ptr().add(kk),
                        out.as_mut_ptr().add(i * n),
                        k,
                        kk_len,
                        n,
                    );
                }
            }
        } else {
            for i in 0..m {
                let a_row = &a[i * k + kk..][..kk_len];
                let out_row = &mut out[i * n..];
                let b_slice = &b_t[kk..];
                for j in (0..n).step_by(4) {
                    let remaining = n - j;
                    let mut s0 = 0.0f32;
                    let mut s1 = 0.0f32;
                    let mut s2 = 0.0f32;
                    let mut s3 = 0.0f32;
                    for t in 0..kk_len {
                        let av = a_row[t];
                        s0 += av * b_slice[j * k + t];
                        if remaining > 1 {
                            s1 += av * b_slice[(j + 1) * k + t];
                        }
                        if remaining > 2 {
                            s2 += av * b_slice[(j + 2) * k + t];
                        }
                        if remaining > 3 {
                            s3 += av * b_slice[(j + 3) * k + t];
                        }
                    }
                    out_row[j] += s0;
                    if remaining > 1 {
                        out_row[j + 1] += s1;
                    }
                    if remaining > 2 {
                        out_row[j + 2] += s2;
                    }
                    if remaining > 3 {
                        out_row[j + 3] += s3;
                    }
                }
            }
        }
        kk = kk_end;
    }
}

pub fn i8_dot_product(a: &[u8], b: &[u8], len: usize) -> i32 {
    if is_x86_feature_detected!("avx2") && len >= 32 {
        unsafe {
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            let chunks = len / 32;
            let mut acc = _mm256_setzero_si256();

            for i in 0..chunks {
                let base = i * 32;

                let a_lo128 = _mm_loadu_si128(a_ptr.add(base) as *const __m128i);
                let a_hi128 = _mm_loadu_si128(a_ptr.add(base + 16) as *const __m128i);
                let b_lo128 = _mm_loadu_si128(b_ptr.add(base) as *const __m128i);
                let b_hi128 = _mm_loadu_si128(b_ptr.add(base + 16) as *const __m128i);

                let a_lo = _mm256_cvtepi8_epi16(a_lo128);
                let a_hi = _mm256_cvtepi8_epi16(a_hi128);
                let b_lo = _mm256_cvtepi8_epi16(b_lo128);
                let b_hi = _mm256_cvtepi8_epi16(b_hi128);

                let mul_lo = _mm256_mullo_epi16(a_lo, b_lo);
                let mul_hi = _mm256_mullo_epi16(a_hi, b_hi);

                acc = _mm256_add_epi32(
                    acc,
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<0>(mul_lo)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(mul_lo)),
                    ),
                );
                acc = _mm256_add_epi32(
                    acc,
                    _mm256_add_epi32(
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<0>(mul_hi)),
                        _mm256_cvtepi16_epi32(_mm256_extracti128_si256::<1>(mul_hi)),
                    ),
                );
            }

            let hi = _mm256_extracti128_si256::<1>(acc);
            let lo = _mm256_castsi256_si128(acc);
            let sum128 = _mm_add_epi32(hi, lo);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(sum128, hi64);
            let hi32 = _mm_shuffle_epi32::<0x01>(sum64);
            let result = _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32));

            let offset = chunks * 32;
            let mut r = result;
            for i in offset..len {
                r += (a[i] as i8 as i32) * (b[i] as i8 as i32);
            }
            r
        }
    } else {
        let mut sum = 0i32;
        for i in 0..len {
            sum += (a[i] as i8 as i32) * (b[i] as i8 as i32);
        }
        sum
    }
}

/// True 1-bit XNOR+popcount. Each byte packs 8 binary values (+1/-1).
/// `popcounts[i]` = number of matching bits in byte i (0..8).
/// For a full dot product: sum = 2 * sum(popcounts) - n_bits.
pub fn xnor_popcount_1bit(a: &[u8], b: &[u8], popcounts: &mut [u32], n_bits: usize) {
    let n_bytes = n_bits.div_ceil(8);
    assert_eq!(a.len(), n_bytes);
    assert_eq!(b.len(), n_bytes);
    assert_eq!(popcounts.len(), n_bytes);

    if is_x86_feature_detected!("avx2") && n_bytes >= 32 {
        unsafe { xnor_popcount_1bit_avx2(a.as_ptr(), b.as_ptr(), popcounts.as_mut_ptr(), n_bytes) };
    } else {
        for i in 0..n_bytes {
            popcounts[i] = (!a[i] ^ b[i]).count_ones();
        }
    }

    // Mask out bits beyond n_bits in the last byte.
    let rem = n_bits % 8;
    if rem != 0 {
        let mask = (1u8 << rem) - 1;
        popcounts[n_bytes - 1] =
            ((!a[n_bytes - 1] ^ b[n_bytes - 1]) & mask).count_ones();
    }
}

/// AVX2 per-byte popcount using VPSHUFB nibble LUT.
/// Each 32 input bytes produce 32 byte popcounts (widened to u32).
#[target_feature(enable = "avx2")]
unsafe fn xnor_popcount_1bit_avx2(a: *const u8, b: *const u8, popcounts: *mut u32, n_bytes: usize) {
    let lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    ));
    let low_mask = _mm256_set1_epi8(0x0F);
    let all_ones = _mm256_set1_epi8(-1);

    let chunks = n_bytes / 32;

    for i in 0..chunks {
        let byte_idx = i * 32;
        let va = _mm256_loadu_si256(a.add(byte_idx) as *const __m256i);
        let vb = _mm256_loadu_si256(b.add(byte_idx) as *const __m256i);
        // XNOR = NOT(XOR(a, b)) — matching bits become 1.
        let xnor = _mm256_andnot_si256(_mm256_xor_si256(va, vb), all_ones);

        let lo = _mm256_and_si256(xnor, low_mask);
        let hi = _mm256_and_si256(_mm256_srli_epi16(xnor, 4), low_mask);
        let counts = _mm256_add_epi8(_mm256_shuffle_epi8(lut, lo), _mm256_shuffle_epi8(lut, hi));

        let lo128 = _mm256_castsi256_si128(counts);
        let hi128 = _mm256_extracti128_si256::<1>(counts);

        _mm256_storeu_si256(
            popcounts.add(byte_idx) as *mut __m256i,
            _mm256_cvtepu8_epi32(lo128),
        );
        _mm256_storeu_si256(
            popcounts.add(byte_idx + 8) as *mut __m256i,
            _mm256_cvtepu8_epi32(_mm_srli_si128(lo128, 8)),
        );
        _mm256_storeu_si256(
            popcounts.add(byte_idx + 16) as *mut __m256i,
            _mm256_cvtepu8_epi32(hi128),
        );
        _mm256_storeu_si256(
            popcounts.add(byte_idx + 24) as *mut __m256i,
            _mm256_cvtepu8_epi32(_mm_srli_si128(hi128, 8)),
        );
    }

    let offset = chunks * 32;
    for i in offset..n_bytes {
        *popcounts.add(i) = (!*a.add(i) ^ *b.add(i)).count_ones();
    }
}

pub fn xnor_popcount_2bit(a: &[u8], b: &[u8], out: &mut [u8], n: usize) {
    if is_x86_feature_detected!("avx2") && n >= 128 {
        unsafe { xnor_popcount_2bit_avx2(a.as_ptr(), b.as_ptr(), out.as_mut_ptr(), n) };
    } else {
        for i in 0..n {
            let byte_a = a[i / 4];
            let byte_b = b[i / 4];
            let shift = (i % 4) * 2;
            let bits_a = (byte_a >> shift) & 0x03;
            let bits_b = (byte_b >> shift) & 0x03;
            let w_a: f32 = match bits_a {
                0x01 => 1.0,
                0x02 => -1.0,
                _ => 0.0,
            };
            let w_b: f32 = match bits_b {
                0x01 => 1.0,
                0x02 => -1.0,
                _ => 0.0,
            };
            if i % 8 == 0 {
                out[i / 8] = 0;
            }
            if w_a * w_b > 0.0 {
                out[i / 8] |= 1 << (i % 8);
            }
        }
    }

    // Mask out bits beyond n in the last output byte.
    let rem = n % 8;
    if rem != 0 {
        let mask = (1u8 << rem) - 1;
        out[n / 8] &= mask;
    }
}
