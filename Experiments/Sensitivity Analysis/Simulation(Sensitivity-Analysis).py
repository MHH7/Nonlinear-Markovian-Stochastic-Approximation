import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pathlib
import webbrowser
import os

# --- KERNEL AND G-FUNCTION DEFINITIONS ---

def g_linear(theta, x):
    """
    A simple linear update function to isolate the effects of the kernel's properties.
    The algorithm seeks the root of E[X] - theta = 0.
    """
    return x - theta

# --- Kernel Factory Functions (Family 1) ---

def make_smooth_kernel_tanh(beta):
    """
    Creates a smooth two-state kernel based on the hyperbolic tangent function.
    
    Args:
        beta (float): The "smoothness dial". Controls the derivative's magnitude
                      at the origin. Must be < 2 for stability.
    """
    def kernel(x, theta):
        mean_val = np.tanh(beta * theta)
        p_switch = np.clip(0.5 - mean_val, 0.0, 1.0)
        return -x if np.random.rand() < p_switch else x
        
    return kernel

def make_nonsmooth_kernel_oscillatory(lam):
    """
    Creates a non-smooth two-state kernel based on an oscillatory function.
    
    Args:
        lam (float): The "non-smoothness dial". Controls the amplitude of the
                     derivative's oscillations. Must be < 1 for stability.
    """
    def kernel(x, theta):
        if theta == 0:
            mean_val = 0.0
        else:
            mean_val = lam * (theta + theta**2 * np.sin(1.0/theta))
            
        p_switch = np.clip(0.5 - mean_val, 0.0, 1.0)
        
        return -x if np.random.rand() < p_switch else x
        
    return kernel

# --- Create the 6 specific kernels for the experiment ---

# Family 1: Smooth Kernels with ~10x and ~100x differences in derivative magnitude
kernel_smooth_low_deriv = make_smooth_kernel_tanh(beta=0.02)  # Deriv ~0.02
kernel_smooth_med_deriv = make_smooth_kernel_tanh(beta=0.2)   # Deriv ~0.2
kernel_smooth_high_deriv = make_smooth_kernel_tanh(beta=1.8)  # Deriv ~1.8

# Family 2: Non-Smooth Kernels with 10x and 90x differences in oscillation amplitude
kernel_nonsmooth_low_osc = make_nonsmooth_kernel_oscillatory(lam=0.01)
kernel_nonsmooth_med_osc = make_nonsmooth_kernel_oscillatory(lam=0.1)
kernel_nonsmooth_high_osc = make_nonsmooth_kernel_oscillatory(lam=0.9)


# --- SIMULATION PARAMETERS ---
alphas = np.array([1e-2, 1e-2/2, 1e-2/4, 1e-2/8, 1e-2/16])
T = 1_000_000
num_trajectories = 1
noise_width = 0.1 # The noise will be uniform in [-0.1, 0.1]

# --- SAVE PATH ---
save_dir = pathlib.Path(
    r"E:\Uni\Research\Stochastic Approximation\Reports\Nonlinear-Markovian-Stochastic-Approximation"
)
pdf_path = save_dir / "smoothness_sensitivity_analysis_large_diff_family1.pdf"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"📂 Save directory: {save_dir}")
print(f"📄 PDF path: {pdf_path}\n")

# --- KERNEL SELECTION ---
kernels = [
    ("Smooth (Low Deriv, β=0.02)",    kernel_smooth_low_deriv),
    ("Smooth (Med Deriv, β=0.2)",    kernel_smooth_med_deriv),
    ("Smooth (High Deriv, β=1.8)",   kernel_smooth_high_deriv),
    ("Non-Smooth (Low Osc, λ=0.01)",  kernel_nonsmooth_low_osc),
    ("Non-Smooth (Med Osc, λ=0.1)",  kernel_nonsmooth_med_osc),
    ("Non-Smooth (High Osc, λ=0.9)", kernel_nonsmooth_high_osc),
]

initial_state = -0.5
theta_star = 0.0

polyak_ruppert_avgs = {name: {} for name, _ in kernels}

# --- MAIN SIMULATION LOOP ---
print("🚀 Starting smoothness sensitivity experiment...")
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
            xi = np.random.uniform(-noise_width, noise_width)
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
    
    plt.suptitle("Smoothness Sensitivity Analysis of SA Kernels (Family 1 - Large Differences)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

if pdf_path.exists():
    print(f"\n✅ PDF saved to: {pdf_path}")
    webbrowser.open(pdf_path.as_uri())
else:
    print("\n❌ Failed to create PDF")

