import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Design choices I made and why:
# 1. I compute the KDE with NumPy broadcasting which is fast enough for typical class data.
#      This avoids loops and avoids needing SciPy, but still follows the math directly.
# 2. I use a default grid of 200 points from (min - 3h) to (max + 3h) so the curve
#      includes tails for smooth kernels like Gaussian.
# 3. I keep the implementation 1D only as that is the standard KDE intro case.

# Helper function: Silverman's rule-of-thumb bandwidth
def silverman_bandwidth(data_values: np.ndarray) -> float:
    """
    Silverman's plug-in bandwidth for 1D KDE:
        h = 0.9 * min(std, IQR/1.34) * n^(-1/5)

    Why this works:
    - It adapts to scale/variation of data.
    - The IQR term is robust to outliers.
    - n^(-1/5) shrinks bandwidth as sample size grows.
    """
    data_values = np.asarray(data_values, dtype=float)
    data_values = data_values[np.isfinite(data_values)]
    n = data_values.size
    if n < 2:
        raise ValueError("Need at least 2 data points for bandwidth calculation.")

    standard_deviation = np.std(data_values, ddof=1)
    q75, q25 = np.percentile(data_values, [75, 25])
    interquartile_range = q75 - q25

    scale = min(standard_deviation, interquartile_range / 1.34) if interquartile_range > 0 else standard_deviation
    if scale <= 0:
        # If all values are identical, KDE is not meaningful as a smooth density.
        # We'll return a tiny bandwidth to avoid division-by-zero errors
        return 1e-6

    bandwidth = 0.9 * scale * (n ** (-1 / 5))
    return float(bandwidth)

# Kernel functions (k(z)) where z = (x - xi) / h
def gaussian_kernel(z: np.ndarray) -> np.ndarray:
    # k(z) = (1/sqrt(2π)) * exp(-z^2/2)
    return (1.0 / np.sqrt(2.0 * np.pi)) * np.exp(-0.5 * z ** 2)


def uniform_kernel(z: np.ndarray) -> np.ndarray:
    # k(z) = 1/2 for |z| <= 1, else 0
    return np.where(np.abs(z) <= 1.0, 0.5, 0.0)


def epanechnikov_kernel(z: np.ndarray) -> np.ndarray:
    # k(z) = (3/4) * (1 - z^2) for |z| <= 1, else 0
    return np.where(np.abs(z) <= 1.0, 0.75 * (1.0 - z ** 2), 0.0)

# Main KDE function (computes the density curve)
def compute_kde_1d(
    data: "pd.Series | np.ndarray",
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    grid_points: int = 200,
    grid: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Computes 1D KDE:
        f_hat(x) = (1/(n*h)) * sum_{i=1..n} k((x - x_i)/h)

    Inputs:
    - data: pandas Series or numpy array (1D numeric)
    - kernel: "gaussian" (default), "uniform", "epanechnikov"
    - bandwidth: if None, use Silverman plug-in bandwidth
    - grid_points: number of x points to evaluate on (ignored if grid is provided)
    - grid: optional custom grid to evaluate KDE on

    Returns:
    - x_grid: points where KDE is evaluated
    - density: KDE values at x_grid
    - bandwidth_used: bandwidth actually used
    """
    # Convert input to clean numpy array
    if isinstance(data, pd.Series):
        data_values = data.to_numpy(dtype=float)
    else:
        data_values = np.asarray(data, dtype=float)

    data_values = data_values[np.isfinite(data_values)]
    n = data_values.size
    if n == 0:
        raise ValueError("No finite numeric values found in the input data.")

    # Choose bandwidth
    bandwidth_used = silverman_bandwidth(data_values) if bandwidth is None else float(bandwidth)
    if bandwidth_used <= 0:
        raise ValueError("Bandwidth must be > 0.")

    # Choose the kernel function
    kernel = kernel.lower().strip()
    if kernel == "gaussian":
        kernel_function = gaussian_kernel
    elif kernel in ["uniform", "bump"]:
        kernel_function = uniform_kernel
    elif kernel == "epanechnikov":
        kernel_function = epanechnikov_kernel
    else:
        raise ValueError("kernel must be 'gaussian', 'uniform'/'bump', or 'epanechnikov'.")

    # Build evaluation grid (x locations)
    if grid is None:
        data_min = data_values.min()
        data_max = data_values.max()
        # Extend by 3 bandwidths to include tails
        left = data_min - 3.0 * bandwidth_used
        right = data_max + 3.0 * bandwidth_used
        x_grid = np.linspace(left, right, grid_points)
    else:
        x_grid = np.asarray(grid, dtype=float)

    # Core KDE computation using NumPy broadcasting 
    # Create a matrix of standardized distances: z_ij = (x_j - x_i)/h
    z = (x_grid[None, :] - data_values[:, None]) / bandwidth_used

    # Apply kernel and average across points, then scale by 1/h
    density = (kernel_function(z).mean(axis=0)) / bandwidth_used

    return x_grid, density, bandwidth_used

# Plot function (Matplotlib)
def plot_kde(
    data: "pd.Series | np.ndarray",
    kernel: str = "gaussian",
    bandwidth: float | None = None,
    grid_points: int = 200,
    label: str | None = None
) -> None:
    x_grid, density, bandwidth_used = compute_kde_1d(
        data=data,
        kernel=kernel,
        bandwidth=bandwidth,
        grid_points=grid_points
    )

    plt.figure()
    plt.plot(x_grid, density, linewidth=2)
    plt.title(f"Custom KDE (kernel={kernel}, bandwidth={bandwidth_used:.4g})")
    plt.xlabel("Value")
    plt.ylabel("Estimated density")
    if label is not None:
        plt.legend([label])
    plt.show()


# Demonstration using a class dataset + comparison to seaborn.kdeplot
# seaborn.kdeplot is only used for comparison below 

current_script_path = Path(__file__).resolve().parent

# spreadsheet that test is being performed upon
data_file_path = current_script_path.parent / "data" / "USA_cars_datasets.csv"

if data_file_path.exists():
    cars_data = pd.read_csv(data_file_path)
    data_series = cars_data["mileage"].dropna()
    series_name = "mileage (USA_cars_datasets.csv)"
else:
    # Fallback so the code still runs even if the CSV isn't present
    rng = np.random.default_rng(42)
    data_series = pd.Series(rng.normal(loc=0, scale=1, size=500))
    series_name = "synthetic normal data (fallback)"

print(f"Using data: {series_name} | n={len(data_series)}")

# Plot our custom KDEs (three kernels)
x_g, d_g, h_g = compute_kde_1d(data_series, kernel="gaussian")
x_u, d_u, h_u = compute_kde_1d(data_series, kernel="uniform")
x_e, d_e, h_e = compute_kde_1d(data_series, kernel="epanechnikov")

plt.figure()
plt.plot(x_g, d_g, linewidth=2, label=f"Custom Gaussian (h={h_g:.4g})")
plt.plot(x_u, d_u, linewidth=2, label=f"Custom Uniform (h={h_u:.4g})")
plt.plot(x_e, d_e, linewidth=2, label=f"Custom Epanechnikov (h={h_e:.4g})")
plt.title("Custom KDE (three kernels)")
plt.xlabel("Value")
plt.ylabel("Estimated density")
plt.legend()
plt.show()

# Compare our Gaussian KDE to seaborn.kdeplot 
plt.figure()
plt.plot(x_g, d_g, linewidth=2, label="Custom Gaussian KDE")

# Seaborn KDE for comparison only
sns.kdeplot(data_series, label="Seaborn kdeplot", linewidth=2)

plt.title("Comparison: Custom KDE vs Seaborn kdeplot")
plt.xlabel("Value")
plt.ylabel("Estimated density")
plt.legend()
plt.show()
