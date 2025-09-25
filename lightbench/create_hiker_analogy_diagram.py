import os

import matplotlib.pyplot as plt
import numpy as np


def create_hiker_analogy_diagram():
    """
    Generates and saves a multi-panel diagram illustrating key optimization challenges.

    The diagram includes separate panels for:
    - A saddle point
    - A narrow ravine (representing ill-conditioning)
    - A sharp cliff (representing a discontinuous gradient)
    """
    # --- Create the plot ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw={"aspect": "equal"})
    fig.suptitle(
        "Challenges in Optimization Landscapes: The Hiker's Analogy",
        fontsize=20,
        y=0.98,
    )

    # --- Panel 1: Saddle Point ---
    ax1 = axes[0]
    x1 = np.linspace(-3, 3, 300)
    y1 = np.linspace(-3, 3, 300)
    X1, Y1 = np.meshgrid(x1, y1)
    Z1 = X1**3 + Y1**3
    levels1 = np.linspace(np.min(Z1), np.max(Z1), 30)
    ax1.contourf(X1, Y1, Z1, levels=levels1, cmap="viridis")
    ax1.contour(X1, Y1, Z1, levels=levels1, colors="black", linewidths=0.5, alpha=0.5)
    ax1.set_title("Saddle Point", fontsize=16)
    ax1.text(0, 0, "★", color="red", fontsize=20, ha="center", va="center")

    # --- Panel 2: Ravine (Ill-Conditioning) ---
    ax2 = axes[1]
    x2 = np.linspace(-3, 3, 300)
    y2 = np.linspace(-3, 3, 300)
    X2, Y2 = np.meshgrid(x2, y2)
    Z2 = 0.1 * X2**2 + 10 * Y2**2  # Rosenbrock-like narrow valley
    levels2 = np.linspace(np.min(Z2), np.max(Z2), 30)
    ax2.contourf(X2, Y2, Z2, levels=levels2, cmap="magma")
    ax2.contour(X2, Y2, Z2, levels=levels2, colors="black", linewidths=0.5, alpha=0.5)
    ax2.set_title("Ravine (Ill-Conditioning)", fontsize=16)
    ax2.text(0, 0, "★", color="cyan", fontsize=20, ha="center", va="center")

    # --- Panel 3: Cliff (Discontinuous Gradient) ---
    ax3 = axes[2]
    x3 = np.linspace(-3, 3, 300)
    y3 = np.linspace(-3, 3, 300)
    X3, Y3 = np.meshgrid(x3, y3)
    Z3 = -Y3  # A simple slope
    Z3[X3 > 0] += 3  # Create a sharp cliff
    levels3 = np.linspace(np.min(Z3), np.max(Z3), 30)
    ax3.contourf(X3, Y3, Z3, levels=levels3, cmap="plasma")
    ax3.contour(X3, Y3, Z3, levels=levels3, colors="black", linewidths=0.5, alpha=0.5)
    ax3.set_title("Cliff (Discontinuous Gradient)", fontsize=16)
    ax3.plot([0, 0], [-3, 3], "r--", lw=2)  # Mark the cliff edge

    # --- Style all plots ---
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout for suptitle

    # --- Save the figure ---
    output_path = "docs/assets/hiker_analogy_diagram.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1, dpi=300)
    print(f"Multi-panel diagram saved to {output_path}")


if __name__ == "__main__":
    create_hiker_analogy_diagram()
