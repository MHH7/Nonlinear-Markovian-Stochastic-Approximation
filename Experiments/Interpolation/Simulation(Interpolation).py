import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pathlib
import webbrowser
import os

# --- G-FUNCTION DEFINITIONS ---

def g_linear(theta, x):
    """
    A linear update function. The algorithm seeks the root of E[X] - theta = 0.
    """
    return x - theta

def g_cubic(theta, x):
    """
    A non-linear update function with cubic and rational terms.
    """
    return -2 * theta + (theta**3) / (1 + theta**2) + x

def g_exp_decay(theta, x):
    """
    A non-linear update function with exponential decay.
    """
    return -theta * np.exp(-theta**2) + x


# --- Two-State Kernel Building Blocks ---

def two_state_sin2cos2_centered(x, theta):
    """
    The SMOOTH endpoint for interpolation. A 2-state kernel on {-0.5, 0.5}.
    The probability of switching states is p(theta) = cos^2(theta).
    """
    p = np.cos(theta)**2
    return -x if np.random.rand() < p else x

def two_state_oscillatory(x, theta):
    """
    The NON-SMOOTH endpoint for interpolation.
    The transition probability function is based on the oscillatory function
    m(theta) = theta + theta^2 * sin(1/theta).
    """
    if theta == 0:
        m_theta = 0.0
    else:
        m_theta = theta + theta**2 * np.sin(1.0/theta)
    
    p = np.clip(0.5 + m_theta, 0.0, 1.0)
    return -x if np.random.rand() < p else x

# --- Interpolated Kernels ---

def make_interpolated_kernel(lam):
    """
    Factory function to create a kernel by interpolating between the smooth
    (sin2cos2) and non-smooth (oscillatory) transition probabilities.
    """
    def interpolated_kernel(x, theta):
        p_smooth = np.cos(theta)**2
        
        if theta == 0:
            m_theta = 0.0
        else:
            m_theta = theta + theta**2 * np.sin(1.0/theta)
        p_non_smooth = np.clip(0.5 + m_theta, 0.0, 1.0)
        
        p_final = (1 - lam) * p_smooth + lam * p_non_smooth
        
        return -x if np.random.rand() < p_final else x
        
    return interpolated_kernel

kernel_lam_000 = make_interpolated_kernel(0.0)
kernel_lam_025 = make_interpolated_kernel(0.25)
kernel_lam_050 = make_interpolated_kernel(0.50)
kernel_lam_075 = make_interpolated_kernel(0.75)
kernel_lam_100 = make_interpolated_kernel(1.0)


# --- SIMULATION PARAMETERS ---
alphas = np.array([2e-2/2, 2e-2/4, 2e-2/8, 2e-2/16])
T = 100_000
num_trajectories = 1

# --- SAVE PATH ---
save_dir = pathlib.Path(
    r"E:\Uni\Research\Stochastic Approximation\Reports\Nonlinear-Markovian-Stochastic-Approximation"
)
pdf_path = save_dir / "interpolation_experiment_polyak_ruppert.pdf"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"📂 Save directory: {save_dir}")
print(f"📄 PDF path: {pdf_path}\n")

# --- KERNEL SELECTION ---
kernels = [
    ("Smooth (λ=0.0)",        kernel_lam_000),
    ("Interpolated (λ=0.25)", kernel_lam_025),
    ("Interpolated (λ=0.5)",  kernel_lam_050),
    ("Interpolated (λ=0.75)", kernel_lam_075),
    ("Non-Smooth (λ=1.0)",    kernel_lam_100),
]

initial_state = -0.5
theta_star = 0.0

polyak_ruppert_avgs = {name: {} for name, _ in kernels}

# --- MAIN SIMULATION LOOP ---
print("🚀 Starting interpolation experiment...")
for name, step_fn in kernels:
    print(f"\n🔍 Kernel: {name}")
    for alpha in alphas:
        print(f"  α = {alpha:.1e}", end=" ", flush=True)
        theta = np.zeros(T)
        x = np.zeros(T)
        
        np.random.seed(42)
        theta[0] = np.random.normal(0, 0.1)
        x[0] = initial_state

        for t in range(T - 1):
            x[t+1] = step_fn(x[t], theta[t])
            xi = np.random.normal(0, 0.1)
            
            # --- SELECT G-FUNCTION ---
            # The experiment is currently set to use g_linear.
            # To change, simply replace 'g_linear' with 'g_cubic' or 'g_exp_decay'.
            g_val = g_linear(theta[t], x[t+1])
            
            theta[t+1] = theta[t] + alpha * (g_val + xi)
        
        pr_avg = np.cumsum(theta) / np.arange(1, T + 1)
        polyak_ruppert_avgs[name][alpha] = pr_avg - theta_star
        
        print("✓", end="", flush=True)

# --- PLOTTING ---
print("\n\n🎨 Generating PDF with plots...")
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(12, 12))
    
    for idx, (name, avgs) in enumerate(polyak_ruppert_avgs.items(), 1):
        ax = plt.subplot(3, 2, idx)
        for alpha, pr_error in avgs.items():
            plot_start_idx = 1000
            ax.semilogy(np.abs(pr_error[plot_start_idx:]), linewidth=1.2, label=f'α={alpha:.1e}')

        ax.set_title(name)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|\bar\theta_n - \theta^*|$")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)
    
    plt.suptitle("Convergence of Interpolated Kernels (Smooth to Oscillatory)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

if pdf_path.exists():
    print(f"\n✅ PDF saved to: {pdf_path}")
    webbrowser.open(pdf_path.as_uri())
else:
    print("\n❌ Failed to create PDF")

