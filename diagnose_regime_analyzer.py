# File: diagnose_regime_analyzer.py

import numpy as np
import matplotlib.pyplot as plt
from gnn_agent.rl_loop.mentor_trigger import AdaptiveBCPDMonitor
import random

print("--- Running Regime Analyzer Diagnostic ---")

# 1. Isolate the data that causes the failure in the unit test
#    This is a linearly decreasing set of 16 data points.
print("1. Generating simulated 'Improving' data...")
np.random.seed(42) # for reproducibility
simulated_data = np.array([5.0 - i * 0.2 + random.uniform(-0.1, 0.1) for i in range(16)])

# 2. Instantiate the monitor with the same settings as the test
print("2. Instantiating the AdaptiveBCPDMonitor...")
monitor = AdaptiveBCPDMonitor(pymc_iterations=2000)

# 3. Call the internal analysis method directly on this data
print("3. Calling the internal _analyze_regime method...")
# This requires the modified mentor_trigger.py to return the posterior distributions
regime, slope_posterior, intercept_posterior = monitor._analyze_regime(simulated_data)

print("\n--- DIAGNOSTIC RESULTS ---")
print(f"Detected Regime: '{regime}'")

# 4. Analyze the posterior distribution of the slope
p05 = np.percentile(slope_posterior, 5)
p95 = np.percentile(slope_posterior, 95)
mean_slope = np.mean(slope_posterior)

print(f"Mean of Slope Posterior: {mean_slope:.4f}")
print(f"5th Percentile of Slope: {p05:.4f}")
print(f"95th Percentile of Slope: {p95:.4f}")

if p95 < 0:
    print("Conclusion: 95th percentile IS less than 0. Should be 'Improving'.")
else:
    print("Conclusion: 95th percentile IS NOT less than 0. This is why it defaults to 'Plateau'.")
print("--------------------------\n")


# 5. Visualize the results
print("5. Generating diagnostic plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Regime Analyzer Diagnostics", fontsize=16)

# Plot 1: Data and Fitted Line
ax1.scatter(np.arange(len(simulated_data)), simulated_data, label='Simulated Loss Data')
# We need to un-standardize the fitted line to plot it on the original data
time_steps = np.arange(len(simulated_data))
epsilon = 1e-9
time_mean = np.mean(time_steps)
time_std = np.std(time_steps) + epsilon
loss_mean = np.mean(simulated_data)
loss_std = np.std(simulated_data) + epsilon
# Calculate the fitted line on the original scale
fit_y = (np.mean(intercept_posterior) + np.mean(slope_posterior) * ((time_steps - time_mean) / time_std)) * loss_std + loss_mean
ax1.plot(time_steps, fit_y, color='red', label=f'Fitted Regression (Slope: {mean_slope:.2f})')
ax1.set_title('Data and Model Fit')
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Loss Value')
ax1.legend()
ax1.grid(True)


# Plot 2: Histogram of the Slope Posterior
ax2.hist(slope_posterior, bins=30, density=True, label='Posterior Distribution')
ax2.axvline(x=0, color='black', linestyle='--', label='Zero Slope')
ax2.axvline(x=p05, color='orange', linestyle=':', label=f'5th Percentile ({p05:.2f})')
ax2.axvline(x=p95, color='red', linestyle=':', label=f'95th Percentile ({p95:.2f})')
ax2.set_title('Posterior Distribution of the Slope')
ax2.set_xlabel('Slope Value')
ax2.set_ylabel('Density')
ax2.legend()
ax2.grid(True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('diagnostic_plot.png')
print("Diagnostic plot saved to 'diagnostic_plot.png'")