import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # Parameter t spans two periods [0, 2]
    t: np.ndarray = np.linspace(0.0, 2.0, 2001)

    # Phase within one period [0, 1], repeated over two periods
    phase: np.ndarray = np.mod(t, 1.0)

    # Function 1 (per period): y = 0.5 * (1 - cos(pi * x)), repeated
    y_cos: np.ndarray = 0.5 * (1.0 - np.cos(np.pi * phase))

    # Function 2 (per period): y = x, repeated
    y_linear: np.ndarray = phase

    # Function 3 (per period): y = arccos(-2*x + 1) / pi, repeated
    y_arccos: np.ndarray = np.arccos(-2.0 * phase + 1.0) / np.pi

    # Insert NaNs at wrap points (where phase jumps from ~1 back to 0) to avoid vertical lines
    wrap_indices = np.where(np.diff(phase) < 0)[0] + 1
    if wrap_indices.size > 0:
        y_cos[wrap_indices] = np.nan
        y_linear[wrap_indices] = np.nan
        y_arccos[wrap_indices] = np.nan

    # Colors: same color for the two linked lines within each subplot
    color_f1 = '#1f77b4'
    color_f2 = '#2ca02c'
    color_f3 = '#ff7f0e'

    # Create three separate subplots (not overlaid)
    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    # Make backgrounds transparent
    fig.patch.set_alpha(0.0)
    for ax in axes:
        ax.set_facecolor('none')

    # Common x ticks for 2 seconds
    xticks = [0.0, 0.5, 1.0, 1.5, 2.0]
    xtick_labels = ['0', '0.5', '1', '1.5', '2']

    # Plot 3: arccos(-2x + 1) / pi (per period) and its complement (same color)
    axes[0].plot(t, y_arccos, linewidth=2.0, color=color_f3)
    axes[0].plot(t, 1.0 - y_arccos, linewidth=2.0, color=color_f3, alpha=0.9)
    axes[0].set_xlim(0.0, 2.0)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_xlabel('t(s)')
    axes[0].set_xticks(xticks, xtick_labels)
    axes[0].set_yticks([0.0, 1.0], ['0', '1'])
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(r"$y = \arccos(-2t + 1)/\pi$")
    # axes[0].legend(loc='best')

    # Plot 2: y = x (per period) and its complement (same color)
    axes[1].plot(t, y_linear, linewidth=2.0, color=color_f2)
    axes[1].plot(t, 1.0 - y_linear, linewidth=2.0, color=color_f2, alpha=0.9)
    axes[1].set_xlim(0.0, 2.0)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_xlabel('t(s)')
    axes[1].set_xticks(xticks, xtick_labels)
    axes[1].set_yticks([0.0, 1.0], ['0', '1'])
    axes[1].grid(True, alpha=0.3)
    axes[1].set_title(r"$y = t$")
    # axes[1].legend(loc='best')


    # Plot 1: 0.5 * (1 - cos(pi * x)) and its complement (same color)
    axes[2].plot(t, y_cos, linewidth=2.0, color=color_f1)
    axes[2].plot(t, 1.0 - y_cos, linewidth=2.0, color=color_f1, alpha=0.9)
    axes[2].set_xlim(0.0, 2.0)
    axes[2].set_ylim(0.0, 1.0)
    axes[2].set_xlabel('t(s)')
    axes[2].set_xticks(xticks, xtick_labels)
    axes[2].set_yticks([0.0, 1.0], ['0', '1'])
    axes[2].grid(True, alpha=0.3)
    axes[2].set_title(r"$y = 0.5\,(1 - \cos(\pi t))$")
    # axes[0].legend(loc='best')

    plt.tight_layout()

    # Save figure alongside the script so it is easy to find (transparent background)
    plt.savefig('plot_two_periods_functions.png', dpi=200, bbox_inches='tight', transparent=True)
    # Show the plot if running interactively
    try:
        plt.show()
    except Exception:
        # In headless environments, just ignore show errors
        pass


if __name__ == '__main__':
    main() 







