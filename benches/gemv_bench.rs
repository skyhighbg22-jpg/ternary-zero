use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use half::f16;
use ternary_zero_core::{pack_ternary_to_u32, ternary_quantize_ste, BitLinear};

static TERNARY_LUT: [f32; 4] = [0.0, 1.0, -1.0, 0.0];

fn bench_pack_ternary(c: &mut Criterion) {
    let mut group = c.benchmark_group("pack_ternary");

    for n in [1024, 2048, 4096, 8192] {
        let weights: Vec<i8> = (0..n)
            .map(|i| match i % 3 {
                0 => 1i8,
                1 => -1i8,
                _ => 0i8,
            })
            .collect();

        group.bench_with_input(BenchmarkId::new("pack", n), &weights, |b, w| {
            b.iter(|| pack_ternary_to_u32(w, n));
        });
    }
    group.finish();
}

fn bench_ternary_quantize(c: &mut Criterion) {
    let mut group = c.benchmark_group("ternary_quantize");

    for n in [1024, 2048, 4096, 8192] {
        let weights: Vec<f16> = (0..n)
            .map(|i| f16::from_f32((i as f32 * 0.001).sin()))
            .collect();

        group.bench_with_input(BenchmarkId::new("quantize", n), &weights, |b, w| {
            b.iter(|| ternary_quantize_ste(w, 0.5));
        });
    }
    group.finish();
}

fn bench_cpu_reference_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_reference_gemv");

    for n in [1024, 2048, 4096] {
        let m = 1;
        let weights: Vec<i8> = (0..m * n)
            .map(|i| match i % 3 {
                0 => 1i8,
                1 => -1i8,
                _ => 0i8,
            })
            .collect();
        let activations: Vec<f16> = (0..n)
            .map(|i| f16::from_f32((i as f32 * 0.001).cos()))
            .collect();

        group.bench_with_input(
            BenchmarkId::new("cpu_gemv", n),
            &(weights, activations),
            |b, (w, a)| {
                b.iter(|| {
                    let mut sum = 0.0f32;
                    for i in 0..n {
                        sum += w[i] as f32 * a[i].to_f32();
                    }
                    sum
                });
            },
        );
    }
    group.finish();
}

fn bench_cpu_packed_gemv(c: &mut Criterion) {
    let mut group = c.benchmark_group("cpu_packed_gemv");

    for n in [1024, 2048, 4096, 8192] {
        let m = 1;
        let weights: Vec<i8> = (0..m * n)
            .map(|i| match i % 3 {
                0 => 1i8,
                1 => -1i8,
                _ => 0i8,
            })
            .collect();
        let packed = pack_ternary_to_u32(&weights, n).unwrap();
        let activations: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001).cos()).collect();

        group.bench_with_input(
            BenchmarkId::new("packed_gemv", n),
            &(packed, activations),
            |b, (pw, act)| {
                b.iter(|| {
                    let packed_cols = n / 16;
                    let mut sum = 0.0f32;
                    for pc in 0..packed_cols {
                        let word = pw[pc];
                        let act_base = pc * 16;
                        for w_idx in 0..16 {
                            let bits = (word >> (w_idx * 2)) & 0b11;
                            sum += TERNARY_LUT[bits as usize] * act[act_base + w_idx];
                        }
                    }
                    sum
                });
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_pack_ternary,
    bench_ternary_quantize,
    bench_cpu_reference_gemv,
    bench_cpu_packed_gemv,
);
criterion_main!(benches);
