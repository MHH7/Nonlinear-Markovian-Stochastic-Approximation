import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pathlib
import webbrowser
import os

# --- KERNEL DEFINITIONS ---

def g_cubic(theta, x):
    """The original non-linear function for the stochastic approximation update."""
    return -2 * theta + (theta**3) / (1 + theta**2) + x

# --- CONTINUOUS-STATE KERNELS (Unchanged) ---
def langevin_step(x, theta, epsilon=0.1):
    return x - (epsilon ** 2) / 2 * (x - theta) + epsilon * np.random.normal()

def osc_step(x, theta, epsilon=0.1):
    if theta == 0:
        m_theta = 0.0
    else:
        m_theta = theta + theta**2 * np.sin(1.0/theta)
    return x - (epsilon ** 2) / 2 * (x + m_theta) + epsilon * np.random.normal()

def ou_uniform_smooth(x, theta, epsilon=0.1, noise_width=1.0):
    return x - (epsilon**2)/2*(x - theta) + epsilon * np.random.uniform(-noise_width, noise_width)

def ou_uniform_sin(x, theta, epsilon=0.1, noise_width=1.0):
    m_theta = 0.5 * abs(np.sin(theta))
    return x - (epsilon**2)/2*(x - m_theta) + epsilon * np.random.uniform(-noise_width, noise_width)

def ou_uniform_sqrtosc(x, theta, epsilon=0.1, noise_width=1.0):
    if theta == 0:
        m_theta = 0.0
    else:
        m_theta = 0.1*(theta + theta**2 * np.sin(1.0/theta))
    return x - (epsilon**2)/2*(x - m_theta) + epsilon * np.random.uniform(-noise_width, noise_width)

# --- DISCRETE-STATE KERNELS (500 States) ---

def discrete_smooth_500(x, theta):
    """
    A 500-state discrete analogue of the Ornstein-Uhlenbeck process.
    The transition probabilities are smooth functions of theta, creating a
    mean-reverting drift towards theta. This ensures a stable root at theta*=0.
    """
    N = 500
    state_range = 2.0
    delta_x = state_range / (N - 1)
    current_index = int(round((x + state_range / 2) / delta_x))
    beta = 1.0
    p_right = 1.0 / (1.0 + np.exp(-beta * (theta - x)))
    if current_index <= 0:
        next_index = current_index + 1 if np.random.rand() < p_right else current_index
    elif current_index >= N - 1:
        next_index = current_index - 1 if np.random.rand() >= p_right else current_index
    else:
        next_index = current_index + 1 if np.random.rand() < p_right else current_index - 1
    return (next_index * delta_x) - state_range / 2

def rw_hinge_500(x, theta):
    """A 500-state random walk with a hinge-like transition probability."""
    N = 500
    p = np.clip(theta, 0.0, 1.0)
    if x == 0:
        return 1 if np.random.rand() < p else 0
    elif x == N-1:
        return N-2 if np.random.rand() >= p else N-1
    else:
        return x+1 if np.random.rand() < p else x-1

def rw_cliposc_500(x, theta):
    """A 500-state random walk with a clipped, oscillatory transition probability."""
    N = 500
    if theta == 0:
        r = 0.0
    else:
        r = theta * np.sqrt(abs(theta)) * np.sin(1.0/theta)
        r = np.clip(r, 0.0, 1.0)
    if x == 0:
        return 1 if np.random.rand() < r else 0
    elif x == N-1:
        return N-2 if np.random.rand() >= r else N-1
    else:
        return x+1 if np.random.rand() < r else x-1

def two_state_sin2cos2_centered(x, theta):
    """A 2-state kernel on {-0.5, 0.5}."""
    p = np.cos(theta)**2
    return -x if np.random.rand() < p else x


# --- SIMULATION PARAMETERS ---
alphas = np.array([2e-2/4, 2e-2/16, 2e-2/64, 2e-2/256])
T = 1_000_000
num_trajectories = 1
LOGGING_THRESHOLD = 1e-5
LOG_WINDOW = 5

# --- SAVE PATH ---
save_dir = pathlib.Path(
    r"E:\Uni\Research\Stochastic Approximation\Reports\Nonlinear-Markovian-Stochastic-Approximation"
)
pdf_path = save_dir / "polyak_ruppert_analysis_final.pdf"
save_dir.mkdir(parents=True, exist_ok=True)

print(f"📂 Save directory: {save_dir}")
print(f"📄 PDF path: {pdf_path}\n")

# --- KERNEL SELECTION ---
kernels = [
    ("Langevin",                 langevin_step),
    ("Oscillatory",              osc_step),
    ("OU_Smooth",                ou_uniform_smooth),
    ("OU_Sin",                   ou_uniform_sin),
    ("OU_SqrtOsc",               ou_uniform_sqrtosc),
    ("Discrete_Smooth_500",      discrete_smooth_500), # Renamed
    ("RW_Hinge500",              rw_hinge_500),
    ("RW_ClipOsc500",            rw_cliposc_500),
    ("2State_Sin2Cos2_Centered", two_state_sin2cos2_centered),
]

initial_states = {
    "Discrete_Smooth_500": 0.0, # Renamed
    "RW_Hinge500": 0.0,
    "RW_ClipOsc500": 0.0,
    "2State_Sin2Cos2_Centered": -0.5,
}

polyak_ruppert_avgs = {name: {} for name, _ in kernels}
detailed_logs = {name: [] for name, _ in kernels}

# --- MAIN SIMULATION LOOP ---
print("🚀 Starting simulation for all kernels...")
for name, step_fn in kernels:
    print(f"\n🔍 Kernel: {name}")
    log_event_counter = 1
    for alpha in alphas:
        print(f"  α = {alpha:.1e}", end=" ", flush=True)
        theta, x, xi_history, g_val_history = (np.zeros(T) for _ in range(4))
        
        np.random.seed(42)
        theta[0] = np.random.normal(0, 0.1)
        x[0] = initial_states.get(name, 0.0)

        for t in range(T - 1):
            x[t+1] = step_fn(x[t], theta[t])
            xi = np.random.normal(0, 0.1)
            g_val = g_cubic(theta[t], x[t+1])
            theta[t+1] = theta[t] + alpha * (g_val + xi)
            xi_history[t], g_val_history[t] = xi, g_val
        
        pr_avg = np.cumsum(theta) / np.arange(1, T + 1)
        polyak_ruppert_avgs[name][alpha] = pr_avg
        
        is_below = np.abs(pr_avg) < LOGGING_THRESHOLD
        consecutive_below = np.convolve(is_below, np.ones(LOG_WINDOW, dtype=int), mode='valid')
        stable_idx_arr = np.where(consecutive_below == LOG_WINDOW)[0]

        if len(stable_idx_arr) > 0:
            first_stable_t = stable_idx_arr[0] + LOG_WINDOW - 1
            is_above = np.abs(pr_avg[first_stable_t:]) > LOGGING_THRESHOLD
            unstable_idx_arr = np.where(is_above)[0]
            
            if len(unstable_idx_arr) > 0:
                instability_t = first_stable_t + unstable_idx_arr[0]
                start_log = max(0, instability_t - LOG_WINDOW)
                end_log = min(T, instability_t + LOG_WINDOW)
                
                detailed_logs[name].append({
                    'id': log_event_counter, 'alpha': alpha, 't': instability_t,
                    'theta_window': theta[start_log:end_log], 'x_window': x[start_log + 1:end_log + 1],
                    'xi_window': xi_history[start_log:end_log], 'g_val_window': g_val_history[start_log:end_log],
                    'pr_avg_window': pr_avg[start_log:end_log]
                })
                log_event_counter += 1
        
        print("✓", end="", flush=True)

# --- PLOTTING ---
print("\n\n🎨 Generating PDF with plots and log pages...")
with PdfPages(pdf_path) as pdf:
    fig = plt.figure(figsize=(16, 12))
    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for idx, (name, avgs) in enumerate(polyak_ruppert_avgs.items(), 1):
        ax = plt.subplot(3, 3, idx)
        for i, (alpha, pr_avg) in enumerate(avgs.items()):
            plot_start_idx = 1000
            line_color = plot_colors[i % len(plot_colors)]
            line, = ax.semilogy(np.abs(pr_avg[plot_start_idx:]), linewidth=1.2, label=f'α={alpha:.1e}', color=line_color)
            
            for log in detailed_logs[name]:
                if log['alpha'] == alpha:
                    log_t = log['t']
                    if log_t >= plot_start_idx:
                        plot_x = log_t - plot_start_idx
                        plot_y = np.abs(pr_avg[log_t])
                        ax.text(plot_x, plot_y, f"{log['id']}", color=line_color, weight='bold',
                                fontsize=8, ha='center', va='center',
                                bbox=dict(boxstyle="circle,pad=0.2", fc='none', ec=line_color, lw=1))
                        break

        ax.set_title(name.replace("_"," "), fontsize=10)
        ax.set_xlabel("Iteration")
        ax.set_ylabel(r"$|\bar\theta_t|$")
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
        ax.legend(fontsize=8)
    
    plt.suptitle("Cubic SA with Polyak-Ruppert Averaging (500-State Kernels)", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    pdf.savefig(fig)
    plt.close(fig)

    # --- SUBSEQUENT PAGES: Detailed Logs ---
    log_fig = plt.figure(figsize=(8.5, 11))
    y_cursor = 0.95
    line_height = 0.015
    log_block_height = (1 + 2 + 1 + 1 + 10 + 1) * line_height

    for name, logs in detailed_logs.items():
        if not logs: continue
        
        kernel_title = f"--- Logs for Kernel: {name.replace('_', ' ')} ---"
        if y_cursor - (3 * line_height) < 0.05:
            pdf.savefig(log_fig); plt.close(log_fig)
            log_fig = plt.figure(figsize=(8.5, 11)); y_cursor = 0.95
        log_fig.text(0.5, y_cursor, kernel_title, ha='center', weight='bold', fontsize=12)
        y_cursor -= (3 * line_height)

        for log in logs:
            if y_cursor - log_block_height < 0.05:
                pdf.savefig(log_fig); plt.close(log_fig)
                log_fig = plt.figure(figsize=(8.5, 11)); y_cursor = 0.95

            title = f"Log Event #{log['id']} (α={log['alpha']:.1e}) - Transition @ t={log['t']}"
            log_text = f"{title}\n"
            log_text += "-"*90 + "\n"
            log_text += "{:<10} {:<12} {:<12} {:<12} {:<12} {:<12}\n".format(
                "Iter (t)", "θ_t", "x_{t+1}", "g(θ,x)", "ξ_t", "avg(θ_t)")
            log_text += "-"*90 + "\n"
            
            for i in range(len(log['theta_window'])):
                if i == LOG_WINDOW: log_text += "--- (Instability Occurs) ---\n"
                t_step = log['t'] - LOG_WINDOW + i
                log_text += "{:<10} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e} {:<12.3e}\n".format(
                    t_step, log['theta_window'][i], log['x_window'][i], 
                    log['g_val_window'][i], log['xi_window'][i], log['pr_avg_window'][i])
            
            log_fig.text(0.05, y_cursor, log_text, family='monospace', verticalalignment='top', fontsize=8)
            y_cursor -= log_block_height

    pdf.savefig(log_fig)
    plt.close(log_fig)

if pdf_path.exists():
    print(f"\n✅ PDF saved to: {pdf_path}")
    webbrowser.open(pdf_path.as_uri())
else:
    print("\n❌ Failed to create PDF")

