"""
Benchmark comparing autodiff (JAX) vs finite differences for predictions.

Run with:
    uv run --all-extras python benchmarks/benchmark_autodiff.py
"""

import time
import numpy as np
import polars as pl
import statsmodels.api as sm
from marginaleffects import predictions, avg_predictions, set_autodiff


def generate_data(
    n_obs: int = 5000, n_predictors: int = 50, seed: int = 42
) -> pl.DataFrame:
    """Generate simulated data."""
    np.random.seed(seed)
    X = np.random.randn(n_obs, n_predictors)
    beta = np.random.randn(n_predictors + 1)
    y = beta[0] + X @ beta[1:] + np.random.randn(n_obs) * 0.5
    data = {"y": y}
    for i in range(n_predictors):
        data[f"x{i}"] = X[:, i]
    return pl.DataFrame(data)


def benchmark(func, model, n_runs: int = 10):
    """Benchmark a function with and without autodiff."""
    results = {"autodiff": [], "finite_diff": []}

    # Warm-up
    set_autodiff(True)
    func(model)
    set_autodiff(False)
    func(model)

    # Benchmark with autodiff
    set_autodiff(True)
    for _ in range(n_runs):
        start = time.perf_counter()
        func(model)
        results["autodiff"].append(time.perf_counter() - start)

    # Benchmark without autodiff
    set_autodiff(False)
    for _ in range(n_runs):
        start = time.perf_counter()
        func(model)
        results["finite_diff"].append(time.perf_counter() - start)

    set_autodiff(None)
    return results


def print_results(name: str, results: dict):
    """Print benchmark results."""
    ad_mean = np.mean(results["autodiff"]) * 1000
    fd_mean = np.mean(results["finite_diff"]) * 1000
    speedup = fd_mean / ad_mean
    print(f"{name:25s}  {ad_mean:6.1f} ms  {fd_mean:6.1f} ms  {speedup:.2f}x")


if __name__ == "__main__":
    # Check JAX
    try:
        from marginaleffects.autodiff import _JAX_AVAILABLE

        if not _JAX_AVAILABLE:
            print("JAX not available")
            exit(1)
    except ImportError:
        print("autodiff module not found")
        exit(1)

    print("Autodiff Benchmark (5000 obs, 50 predictors)")
    print("=" * 60)
    print(f"{'Function':25s}  {'JAX':>8s}  {'Finite':>8s}  Speedup")
    print("-" * 60)

    data = generate_data()
    formula = "y ~ " + " + ".join([f"x{i}" for i in range(50)])

    # OLS
    ols = sm.OLS.from_formula(formula, data.to_pandas()).fit()
    print_results("OLS predictions()", benchmark(predictions, ols))
    print_results("OLS avg_predictions()", benchmark(avg_predictions, ols))

    # GLM
    glm = sm.GLM.from_formula(
        formula, data.to_pandas(), family=sm.families.Gaussian()
    ).fit()
    print_results("GLM predictions()", benchmark(predictions, glm))
    print_results("GLM avg_predictions()", benchmark(avg_predictions, glm))

    print("=" * 60)
