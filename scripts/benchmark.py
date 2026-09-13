"""Benchmark inference latency: Pickle (scikit-learn) vs ONNX Runtime."""

import pickle
import time

import numpy as np
import onnxruntime as ort

from prodml.config import get_settings
from prodml.data import clean_data, load_data, split_data
from prodml.export import export_baseline
from prodml.features import prepare_features


def run_benchmark(n_samples: int = 500, n_batch_trials: int = 50) -> None:
    settings = get_settings()
    pkl_path = settings.model_path
    onnx_path = pkl_path.with_suffix(".onnx")

    if not onnx_path.exists():
        print("Exporting ONNX model first...")
        export_baseline(pkl_path, onnx_path)

    # 1. Load models
    with open(pkl_path, "rb") as f:
        pipeline = pickle.load(f)

    model = pipeline.named_steps["model"]
    dv = pipeline.named_steps["vectorizer"]

    session = ort.InferenceSession(str(onnx_path))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 2. Load dataset and select validation rows
    print(f"Loading data from {settings.data_path}...")
    raw_df = load_data(settings.data_path)
    cleaned_df = clean_data(raw_df)
    _, X_val_df, _, _, _, _ = split_data(cleaned_df)

    val_subset = X_val_df.iloc[:n_samples]
    prepared = prepare_features(val_subset)
    X_mat = dv.transform(prepared).toarray().astype(np.float32)

    # 3. Warmup
    for _ in range(5):
        _ = model.predict(X_mat[:5])
        _ = session.run([output_name], {input_name: X_mat[:5]})

    # 4. Single-row latency benchmark
    print(f"\nRunning single-row benchmark across {n_samples} samples...")
    pkl_single_latencies = []
    for i in range(n_samples):
        row = X_mat[i : i + 1]
        t0 = time.perf_counter()
        _ = model.predict(row)
        pkl_single_latencies.append((time.perf_counter() - t0) * 1000)

    onnx_single_latencies = []
    for i in range(n_samples):
        row = X_mat[i : i + 1]
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: row})
        onnx_single_latencies.append((time.perf_counter() - t0) * 1000)

    # 5. Batch latency benchmark (full batch of n_samples)
    print(
        f"Running batch benchmark ({n_samples} rows per call, {n_batch_trials} trials)..."
    )
    pkl_batch_latencies = []
    onnx_batch_latencies = []
    for _ in range(n_batch_trials):
        t0 = time.perf_counter()
        _ = model.predict(X_mat)
        pkl_batch_latencies.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: X_mat})
        onnx_batch_latencies.append((time.perf_counter() - t0) * 1000)

    # 6. Format and display results
    pkl_s_mean = np.mean(pkl_single_latencies)
    pkl_s_p95 = np.percentile(pkl_single_latencies, 95)
    onnx_s_mean = np.mean(onnx_single_latencies)
    onnx_s_p95 = np.percentile(onnx_single_latencies, 95)
    single_speedup = pkl_s_mean / onnx_s_mean

    pkl_b_mean = np.mean(pkl_batch_latencies)
    pkl_b_p95 = np.percentile(pkl_batch_latencies, 95)
    onnx_b_mean = np.mean(onnx_batch_latencies)
    onnx_b_p95 = np.percentile(onnx_batch_latencies, 95)
    batch_speedup = pkl_b_mean / onnx_b_mean

    print("\n" + "=" * 70)
    print(f"{'LATENCY BENCHMARK RESULTS (Pickle vs ONNX Runtime)':^70}")
    print("=" * 70)
    print(
        f"{'Inference Mode':<18} | {'Metric':<10} | {'Pickle':<12} | {'ONNX':<12} | {'Speedup':<10}"
    )
    print("-" * 70)
    print(
        f"{'Single-Row (500x)':<18} | {'Mean':<10} | {pkl_s_mean:8.3f} ms | {onnx_s_mean:8.3f} ms | {single_speedup:8.1f}x"
    )
    print(
        f"{'':<18} | {'P95':<10} | {pkl_s_p95:8.3f} ms | {onnx_s_p95:8.3f} ms | {pkl_s_p95 / onnx_s_p95:8.1f}x"
    )
    print("-" * 70)
    print(
        f"{'Batch (500 rows)':<18} | {'Mean':<10} | {pkl_b_mean:8.3f} ms | {onnx_b_mean:8.3f} ms | {batch_speedup:8.1f}x"
    )
    print(
        f"{'':<18} | {'P95':<10} | {pkl_b_p95:8.3f} ms | {onnx_b_p95:8.3f} ms | {pkl_b_p95 / onnx_b_p95:8.1f}x"
    )
    print("=" * 70 + "\n")


if __name__ == "__main__":
    run_benchmark()
