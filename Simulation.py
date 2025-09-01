# --- Core Libraries ---
# Imports for numerical operations, plotting, and file system management.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pathlib
import webbrowser
import os

# --- Stochastic Approximation (SA) Update Function ---
def g_linear(theta, x):
    """
    Defines the update rule for the SA algorithm. The algorithm seeks the
    root of the equation E[X] - theta = 0.
    """
    return x - theta

# --- Markov Transition Kernels ---
# This section defines the four different Markov transition kernels for the experiments.
# Each kernel is a two-state chain that determines the next state X_{t+1} based on
# the current state X_t and the parameter theta_t.

# Kernel 1: Linear backbone with mild, localized oscillations.
def original_oscillatory_kernel(x, theta):
    """
    A non-smooth kernel with oscillations localized near the origin.
    The mean function is m(theta) = 0.9 * (theta + theta^2 * sin(1/theta)).
    """
    if theta == 0:
        m_theta = 0.0
    else:
        lam = 0.9
        m_theta = lam * (theta + theta**2 * np.sin(1.0/theta))
    
    # Converts the mean function to a switching probability for a two-state chain.
    p = np.clip(0.5 + m_theta, 0.0, 1.0)
    return -x if np.random.rand() < p else x

# Kernel 2: Linear backbone with wild, global oscillations.
def linear_backbone_high_amplitude_kernel(x, theta):
    """
    A kernel with a simple linear global shape but with globally-distributed,
    high-amplitude, non-Lipschitz derivatives.
    """
    # Defines the parameters that control the kernel's shape and texture.
    lam = 0.7
    C = 0.4
    alpha_mod = 0.1
    k_freq = 5.0 * np.pi

    # Creates a symmetric, periodic non-smooth function.
    abs_theta = np.abs(theta)
    s_theta = abs_theta - np.floor(abs_theta)
    if s_theta < 1e-9:
        h_abs = 0.0
    else:
        h_abs = s_theta**2 * np.sin(k_freq / s_theta)
    h_theta = np.sign(theta) * h_abs

    # Modulates the amplitude of the oscillations.
    A_theta = C * (np.exp(-alpha_mod * theta**2) + 1)

    # Combines the linear backbone with the pathological part.
    m_theta = (1 - lam) * theta + A_theta * h_theta

    # Converts the mean function to a switching probability.
    p_theta = (1.0 - m_theta) / 2.0
    p_clipped = np.clip(p_theta, 0.0, 1.0)
    return -x if np.random.rand() < p_clipped else x

# Kernel 3: Sublinear backbone with a sharp two-phase transition and wild oscillations.
def sharp_two_phase_sublinear_kernel(x, theta):
    """
    A kernel with a sublinear global shape. It has a max-growth linear phase
    for |x| < 1e-4, which sharply transitions to a flattening phase.
    """
    # Defines the parameters for the two-phase backbone.
    A = 0.99
    epsilon = 1e-4

    # Defines the parameters for the pathological (oscillating) part.
    C = 1e-7
    alpha_mod = 0.1
    k_freq = 5.0 * np.pi

    # Creates a symmetric, periodic non-smooth function.
    abs_theta = np.abs(theta)
    s_theta = abs_theta - np.floor(abs_theta)
    if s_theta < 1e-9:
        h_abs = 0.0
    else:
        h_abs = s_theta**2 * np.sin(k_freq / s_theta)
    h_theta = np.sign(theta) * h_abs

    # Modulates the amplitude of the oscillations.
    A_theta = C * (np.exp(-alpha_mod * theta**2) + 1)
    
    # Creates the two-phase backbone using a hyperbolic tangent.
    backbone = A * epsilon * np.tanh(theta / epsilon)

    # Combines the sublinear backbone with the pathological part.
    m_theta = backbone + A_theta * h_theta

    # Converts the mean function to a switching probability.
    p_theta = (1.0 - m_theta) / 2.0
    p_clipped = np.clip(p_theta, 0.0, 1.0)
    return -x if np.random.rand() < p_clipped else x

# Kernel 4: Sublinear backbone with a multi-stage transition and wild oscillations.
def sublinear_backbone_high_amplitude_5pi_kernel(x, theta):
    """
    A kernel with a sublinear, multi-stage backbone and a high-amplitude
    pathological part with a frequency of 5*pi.
    """
    # Defines the parameters for the multi-stage shaping backbone.
    S_slow = 0.2
    H1 = 6e-5
    W1 = 1e-4
    H2 = 1e-3
    W2 = 1e-2

    # Defines the parameters for the pathological foundation.
    C = 0.5
    k_freq = 5.0 * np.pi

    # Creates a symmetric, periodic non-smooth function.
    abs_theta = np.abs(theta)
    s_theta = abs_theta - np.floor(abs_theta)
    if s_theta < 1e-9:
        h_abs = 0.0
    else:
        h_abs = s_theta**2 * np.sin(k_freq / s_theta)
    h_theta = np.sign(theta) * h_abs

    # Creates the multi-stage shaping function.
    shaping_function = (H1 * np.tanh(theta / W1) + 
                        H2 * np.tanh(theta / W2) + 
                        S_slow * theta)

    # Combines the shaping function with the pathological part.
    m_theta = shaping_function + C * h_theta

    # Converts the mean function to a switching probability.
    p_theta = (1.0 - m_theta) / 2.0
    p_clipped = np.clip(p_theta, 0.0, 1.0)
    return -x if np.random.rand() < p_clipped else x


# --- Simulation Configuration ---
# Defines the set of stepsizes (alpha values) to test.
alphas = np.array([4e-2, 2e-2, 1e-2, 1e-2 / 2, 1e-2 / 4, 1e-2 / 8])
#, 1e-2 / 16, 1e-2 / 32

# Total number of iterations for each simulation run.
T = 700_000_000
num_trajectories = 1
# Frequency for recording data points to keep memory usage low.
SAMPLING_RATE = T // 50_000 

# --- File Output Settings ---
# Sets the directory where the output PDF will be saved.
save_dir = pathlib.Path(
    r"E:\Uni\Research\Stochastic Approximation\Reports\Nonlinear-Markovian-Stochastic-Approximation"
)
save_dir.mkdir(parents=True, exist_ok=True)

# --- Experiment Selection ---
# A dictionary to hold all implemented kernels for easy selection.
all_kernels = {
    "Original Oscillatory": original_oscillatory_kernel,
    "Linear Backbone High Amplitude (5pi)": linear_backbone_high_amplitude_kernel,
    "Sublinear Sharp Two-Phase": sharp_two_phase_sublinear_kernel,
    "Sublinear Multi-Stage High-Amp": sublinear_backbone_high_amplitude_5pi_kernel
}

# Select which kernel to run for the experiment by changing this string.
kernel_name_to_run = "Linear Backbone High Amplitude (5pi)"
kernel_to_run = (kernel_name_to_run, all_kernels[kernel_name_to_run])

# Dynamically generate the output filename based on the selected kernel.
pdf_filename = f"stepsize_vs_bias_{kernel_name_to_run.replace(' ', '_').lower()}.pdf"
pdf_path = save_dir / pdf_filename

print(f"📂 Save directory: {save_dir.resolve()}")
print(f"📄 PDF path: {pdf_path.resolve()}\n")

# --- Simulation Initialization ---
# Defines the initial state of the Markov chain.
initial_state = -0.5
# Defines the true root of the problem.
theta_star = 0.0
# A dictionary to store the results for each stepsize alpha.
polyak_ruppert_avgs = {alpha: None for alpha in alphas}

# --- Main Simulation Loop ---
print("🚀 Starting stepsize vs. bias experiment...")
name, step_fn = kernel_to_run
print(f"\n🔍 Kernel: {name}")

# Iterates through each stepsize alpha to run an experiment.
for alpha in alphas:
    print(f"  α = {alpha:.2e}", end=" ", flush=True)
    
    # Initializes variables for the current run.
    pr_errors_for_plot = []
    np.random.seed(42)
    theta_current = 0.001
    x_current = initial_state
    theta_sum = theta_current
    
    # This loop performs the main Stochastic Approximation updates.
    for t in range(T - 1):
        # Generate the next state from the Markov kernel.
        x_next = step_fn(x_current, theta_current)
        # Generate i.i.d. Rademacher noise.
        xi = 1.0 if np.random.rand() < 0.5 else -1.0
        # Calculate the g-function value.
        g_val = g_linear(theta_current, x_next)
        # Perform the SA update.
        theta_next = theta_current + alpha * (g_val + xi)
        
        # Update state variables for the next iteration.
        theta_current = theta_next
        x_current = x_next
        # Maintain a running sum of theta for Polyak-Ruppert averaging.
        theta_sum += theta_current
        
        # Periodically record the Polyak-Ruppert average to save memory.
        if (t + 1) % SAMPLING_RATE == 0:
            pr_avg = theta_sum / (t + 2)
            pr_errors_for_plot.append(pr_avg - theta_star)

    # Store the final, downsampled list of errors for this alpha.
    polyak_ruppert_avgs[alpha] = np.array(pr_errors_for_plot)
    print("✓", end="", flush=True)

# --- Plotting Results ---
print("\n\n🎨 Generating PDF with plot and annotations...")
# Creates a PDF file to save the final plot.
with PdfPages(pdf_path) as pdf:
    # Sets up the figure and axes for the plot.
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    # Loops through the results for each alpha and plots the error trajectory.
    for i, (alpha, pr_error) in enumerate(polyak_ruppert_avgs.items()):
        plot_start_idx = max(1, 1000 // SAMPLING_RATE)
        y_data = np.abs(pr_error[plot_start_idx:])
        x_indices = (np.arange(len(y_data)) + plot_start_idx) * SAMPLING_RATE
        line, = ax.semilogy(x_indices, y_data, linewidth=1.2, label=f'α={alpha:.2e}', color=plot_colors[i])
        
        # Annotates the final point of each trajectory with its value.
        if len(x_indices) > 0 and len(y_data) > 0:
            final_x = x_indices[-1]
            final_y = y_data[-1]
            ax.text(final_x, final_y, f' {final_y:.2e}', 
                    color=line.get_color(), fontsize=8, verticalalignment='center')

    # Configures plot titles, labels, grid, and legend.
    ax.set_title(f"Effect of Stepsize on Bias ({name})")
    ax.set_xlabel("Iteration (n)")
    ax.set_ylabel(r"$|\bar\theta_n - \theta^*|$")
    ax.grid(True, which="both", linestyle="--", alpha=0.6)
    ax.legend(fontsize=9)
    
    # Saves the figure to the PDF.
    plt.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)

# --- Finalization ---
# Opens the generated PDF file if the script was successful.
if pdf_path.exists():
    print(f"\n✅ PDF saved to: {pdf_path.resolve()}")
    webbrowser.open(pdf_path.as_uri())
else:
    print("\n❌ Failed to create PDF")