pub fn f32_add(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] + b[i];
    }
}

pub fn f32_sub(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] - b[i];
    }
}

pub fn f32_mul(a: &[f32], b: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
}

pub fn f32_scale(a: &[f32], scale: f32, out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i] * scale;
    }
}

pub fn f32_dot(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    for i in 0..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

pub fn f32_sum(a: &[f32]) -> f32 {
    a.iter().sum()
}

pub fn f32_max(a: &[f32]) -> f32 {
    a.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
}

pub fn f32_exp(a: &[f32], out: &mut [f32]) {
    for i in 0..a.len() {
        out[i] = a[i].exp();
    }
}

pub fn f32_matmul_row(a: &[f32], b_t: &[f32], out_row: &mut [f32], k: usize, n: usize) {
    for j in 0..n {
        let mut sum = 0.0f32;
        for t in 0..k {
            sum += a[t] * b_t[j * k + t];
        }
        out_row[j] = sum;
    }
}

pub fn f32_matmul(a: &[f32], b_t: &[f32], out: &mut [f32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        f32_matmul_row(&a[i * k..], b_t, &mut out[i * n..], k, n);
    }
}

pub fn i8_dot_product(a: &[u8], b: &[u8], len: usize) -> i32 {
    let mut sum = 0i32;
    for i in 0..len {
        sum += (a[i] as i8 as i32) * (b[i] as i8 as i32);
    }
    sum
}

pub fn xnor_popcount_1bit(a: &[u8], b: &[u8], popcounts: &mut [u32], n_bits: usize) {
    let n_bytes = (n_bits + 7) / 8;
    assert_eq!(a.len(), n_bytes);
    assert_eq!(b.len(), n_bytes);
    assert_eq!(popcounts.len(), n_bytes);
    for i in 0..n_bytes {
        popcounts[i] = (!a[i] ^ b[i]).count_ones();
    }
}

pub fn xnor_popcount_2bit(a: &[u8], b: &[u8], out: &mut [u8], n: usize) {
    for i in 0..n {
        let byte_a = a[i / 2];
        let byte_b = b[i / 2];
        let bits_a = if i % 2 == 0 {
            byte_a & 0x03
        } else {
            (byte_a >> 4) & 0x03
        };
        let bits_b = if i % 2 == 0 {
            byte_b & 0x03
        } else {
            (byte_b >> 4) & 0x03
        };

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

        let dot = w_a * w_b;
        if i % 8 == 0 {
            out[i / 8] = 0;
        }
        if dot > 0.0 {
            out[i / 8] |= 1 << (i % 8);
        }
    }
}

pub fn detect_simd_info() -> &'static str {
    "scalar"
}
