import numpy as np
import os
import time  # For measuring runtime
from multiprocessing import Pool
import matplotlib.pyplot as plt
import itertools

# -------------------------------
# Model Functions and Loss
# -------------------------------

# Model for all solar cycles
def solar_model(t, params):
    if len(params) != 30:
        raise ValueError("Expected 30 parameters (10 cycles), got {len(params)}.")
    x = np.zeros_like(t)
    for k in range(10):  # 10 cycles
        T0_k, Ts_k, Td_k = params[3 * k: 3 * k + 3]

        # Enforce minimum thresholds for stability
        Ts_k = max(Ts_k, 1e-6)  # Avoid division by zero
        Td_k = max(Td_k, 1e-6)  # Avoid division by zero

        if k < 9:
            T0_next = params[3 * (k + 1)]
            mask = (t >= T0_k) & (t < T0_next)
        else:
            mask = (t >= T0_k)
        x[mask] += ((t[mask] - T0_k) / Ts_k) ** 2 * np.exp(-((t[mask] - T0_k) / Td_k) ** 2)
    return x

# Loss function (MSE)
def mse(params):
    predictions = solar_model(time_data, params)
    return np.mean((sn - predictions) ** 2)

# -------------------------------
# Simulated Annealing Algorithm
# -------------------------------

def simulated_annealing(x0, T0, sigma, f, n_iter=20000):
    x = x0.copy()
    T = T0
    n_params = len(x0)
    means = np.zeros(n_params)
    cov_matrix = np.diag(np.full(n_params, sigma))
    mse_values = []

    for iter_counter in range(n_iter):
        x_old = x.copy()
        x_proposal = x_old + np.random.multivariate_normal(means, cov_matrix)
        x_proposal = np.clip(x_proposal, a_min=0, a_max=np.inf)  # Enforce positivity
        DeltaE = f(x_proposal) - f(x_old)

        if np.exp(-np.clip(DeltaE / T, -100, 100)) >= np.random.rand():
            x = x_proposal
        else:
            x = x_old

        T = T0 * (1 - iter_counter / n_iter)
        mse_values.append(f(x))

    return x, mse_values

# -------------------------------
# Hyperparameter Optimization
# -------------------------------

# Parallelized function for hyperparameter optimization
def hyperparameter_task(params):
    T0, sigma = params
    print(f"Testing T0={T0}, sigma={sigma}")
    outSA, mse_history = simulated_annealing(x0, T0, sigma, f=mse, n_iter=20000)
    final_loss = mse(outSA)

    # Save MSE evolution plot
    plt.figure()
    plt.plot(mse_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title(f"MSE Evolution (T0={T0}, sigma={sigma})")
    plt.grid()
    plt.savefig(f"mse_T0_{T0}sigma{sigma}.png")
    plt.close()

    return T0, sigma, final_loss

# -------------------------------
# Model Calibration
# -------------------------------

# Parallelized function for independent SA runs
def calibration_task(x0_noise):
    params, _ = simulated_annealing(x0_noise, best_T0, best_sigma, f=mse, n_iter=20000)
    return params

# -------------------------------
# Main Script for Cluster
# -------------------------------

if __name__ == "__main__":
    # Start time measurement
    total_start_time = time.time()

    # Load data
    data = np.loadtxt("data_team8.csv", delimiter=",", skiprows=1)
    time_data = data[:, 0]
    sn = data[:, 1]

    # Initial parameters
    initial_T0 = [1823.3, 1833.9, 1843.5, 1856.0, 1867.2, 1878.9, 1889.6, 1901.7, 1913.6, 1923.6]
    initial_Ts = [0.3] * 10
    initial_Td = [5.0] * 10
    x0 = []
    for i in range(10):
        x0.extend([initial_T0[i], initial_Ts[i], initial_Td[i]])
    x0 = np.array(x0)

    # Hyperparameter grid
    T0_values = np.linspace(0, 10, 10)
    sigma_values = np.linspace(10**-9, 10**-6, 10)
    hyperparameters = list(itertools.product(T0_values, sigma_values))

    # Task 1: Hyperparameter Optimization
    with Pool(32) as pool:
        results = pool.map(hyperparameter_task, hyperparameters)
    results_sorted = sorted(results, key=lambda x: x[2])
    best_T0, best_sigma, _ = results_sorted[0]
    print(f"Best Hyperparameters: T0={best_T0}, Sigma={best_sigma}")

    # Task 2: Model Calibration
    noisy_initial_conditions = [x0 + np.random.normal(0, 0.1, size=30) for _ in range(10)]
    with Pool(10) as pool:
        calibration_results = pool.map(calibration_task, noisy_initial_conditions)

    # Combine results: Center of Mass
    calibration_results = np.array(calibration_results)
    center_of_mass = np.mean(calibration_results, axis=0)
    print("Final Optimized Parameters (Center of Mass):", center_of_mass)

    # End time measurement
    total_runtime = time.time() - total_start_time
    print(f"Total Runtime: {total_runtime:.2f} seconds")

    # Save the final fit
    predictions = solar_model(time_data, center_of_mass)
    plt.figure()
    plt.plot(time_data, sn, label="Original Data", color="orange")
    plt.plot(time_data, predictions, label="Optimized Fit", color="blue")
    plt.xlabel("Year")
    plt.ylabel("SN")
    plt.legend()
    plt.title("Final Optimized Fit")
    plt.grid()
    plt.savefig("final_optimized_fit.png")
    plt.show()

    # Combine results: MSE values from all calibration runs
    mse_calibration_results = [mse(params) for params in calibration_results]
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse_calibration_results)), mse_calibration_results, marker='o', label="Calibration MSE")
    plt.xlabel("Run Index")
    plt.ylabel("Final MSE")
    plt.title("Final MSE for Each Calibration Run")
    plt.grid()
    plt.legend()
    plt.savefig("calibration_mse_summary.png")
    plt.show()

    # Linear correlation between T_S and T_d
    Ts = center_of_mass[1::3]
    Td = center_of_mass[2::3]
    slope, intercept = np.polyfit(Ts, Td, 1)
    Td_fit = slope * Ts + intercept

    plt.figure()
    plt.scatter(Ts, Td, label="T_S vs T_d", color="orange")
    plt.plot(Ts, Td_fit, label=f"Linear Fit: slope={slope:.2f}, intercept={intercept:.2f}", color="blue")
    plt.xlabel("T_S")
    plt.ylabel("T_d")
    plt.title("Linear Correlation Between T_S and T_d")
    plt.legend()
    plt.grid()
    plt.savefig("linear_correlation_Ts_Td.png")
    plt.show()