import numpy as np
import matplotlib.pyplot as plt

# Define nonlinear g functions with corrected drifts
def g_cubic(theta, x):
    return theta**3 / (1 + theta**2) - x  # E[g(0,X)] = -E[X] = 0

def g_tanh(theta, x):
    return np.tanh(theta) - np.tanh(x)  # E[g(0,X)] = 0

def g_exp_decay(theta, x):
    return np.exp(-np.abs(theta)) * (theta - x)  # E[g(0,X)] = -E[X] = 0

g_functions = {
    "Cubic": g_cubic,
    "Tanh": g_tanh,
    "ExpDecay": g_exp_decay
}

# Langevin dynamics
def langevin_step(x, theta, step_size=0.1):
    return x - step_size * (x - theta) + np.sqrt(2 * step_size) * np.random.normal()

# Simulate E[g(theta, X)] under stationarity
def simulate_stationary_expectation(theta, g_func, num_samples=10000, burn_in=1000):
    x = theta  # Start near the stationary mean
    for _ in range(burn_in):
        x = langevin_step(x, theta)
    g_sum = 0.0
    for _ in range(num_samples):
        x = langevin_step(x, theta)
        g_sum += g_func(theta, x)
    return g_sum / num_samples

# Parameters
theta_star = 0.0
alphas = np.array([1e-4, 1e-5, 1e-6])  # Adjusted to feasible α given computational limits
num_trajectories = 5  # Reduced for speed; increase if possible
results = {gname: {'bias1': [], 'bias2': []} for gname in g_functions}

for gname, g_func in g_functions.items():
    for alpha in alphas:
        T = int(1 / alpha)  # Scale steps inversely with α
        burn_in_sa = int(0.2 * T)  # 20% burn-in
        
        bias1_traj, bias2_traj = [], []
        for traj in range(num_trajectories):
            np.random.seed(42 + traj)
            
            # Initialize arrays to track θ and X over time
            theta_vals = np.zeros(T)
            x_vals = np.zeros(T)
            theta_vals[0] = 1.0  # Initial θ
            x_vals[0] = 0.0       # Initial X
            
            # SA Loop
            for t in range(T - 1):
                x_vals[t + 1] = langevin_step(x_vals[t], theta_vals[t])
                xi = np.random.normal(0, 0.1)
                g_val = g_func(theta_vals[t], x_vals[t + 1])
                theta_vals[t + 1] = theta_vals[t] + alpha * (g_val + xi)
            
            # Compute θ_inf as the mean of post-burn-in θ values
            theta_inf = np.mean(theta_vals[burn_in_sa:])
            
            # Estimate biases
            E_g = simulate_stationary_expectation(theta_inf, g_func)
            bias1_traj.append(np.abs(E_g))
            bias2_traj.append(np.abs(theta_inf - theta_star))
        
        results[gname]['bias1'].append(np.mean(bias1_traj))
        results[gname]['bias2'].append(np.mean(bias2_traj))

# Plotting
plt.figure(figsize=(12, 5))
for gname in g_functions:
    plt.loglog(alphas, results[gname]['bias1'], 'o--', label=f'{gname} (Bias1)')
    plt.loglog(alphas, results[gname]['bias2'], 's--', label=f'{gname} (Bias2)')

plt.xlabel(r'$\alpha$')
plt.ylabel('Bias (log scale)')
plt.title('Bias Scaling with Adjusted Iterations')
plt.legend()
plt.grid(True)
plt.show()